// FJC patch (8): duplex mode for CoreAudio, for CamillaDSP 4.1.3 and 5.0.
//
// Capture, processing and playback inside the IO callback of ONE audio unit, for a config that
// captures from and plays to the same device with no resampling and no rate adjust. There are no
// helper threads and no queues between them: each callback pulls the input of that same IO cycle,
// runs the pipeline and fills the output. The round trip is the device latency plus one buffer,
// instead of three buffers plus `target_level`.
//
// CAMILLADSP_DUPLEX=1 turns it on and =0 off; without the variable, DUPLEX_BY_DEFAULT decides. A
// config it cannot run (two devices, resampling, rate adjust) uses the regular threads. `chunksize`
// must be what the device grants per buffer. A skipped IO cycle is logged with the same words as an
// underrun of the regular backend ("Playback interrupted"), so the same tools count it.
//
// This file is installed as src/coreaudio_backend/duplex.rs by camilladsp_bufer_coreaudio.py
// (fj-acoustics-studio, 19/11/2026), which also adds the hook where the threads are started. It is a
// template for both versions: a line that ends in a version tag is kept, without the tag, only for
// that version.

use std::os::raw::c_void;
use std::ptr::{NonNull, null};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use coreaudio::audio_unit::audio_format::LinearPcmFlags;
use coreaudio::audio_unit::macos_helpers::{AliveListener, RateListener, set_device_sample_rate};
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::{AudioUnit, Element, IOType, SampleFormat, Scope, StreamFormat};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded};
use objc2_audio_toolbox::{
    AudioUnitRender, AudioUnitRenderActionFlags, kAudioOutputUnitProperty_CurrentDevice,
    kAudioOutputUnitProperty_EnableIO, kAudioUnitProperty_StreamFormat,
};
use objc2_core_audio::{
    AudioDeviceID, AudioObjectGetPropertyData, AudioObjectPropertyAddress,
    AudioObjectSetPropertyData, kAudioDevicePropertyBufferFrameSize,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
};
use objc2_core_audio_types::{AudioBuffer, AudioBufferList};
use parking_lot::RwLock;

use super::device::get_device_id_from_name_and_scope;
use crate::audiochunk::ChunkStats;
use crate::config::{self, BinarySampleFormat};
use crate::filters::fftconv::ConvCoeffCache;
use crate::pipeline::Pipeline;
use crate::processing::PipelineConfig;
use crate::utils::conversions::{buffer_to_chunk_rawbytes, chunk_to_buffer_rawbytes};
use crate::{
    CaptureStatus, CommandMessage, PlaybackStatus, ProcessingParameters, ProcessingState,
    StatusMessage,
};

/// What a build does when CAMILLADSP_DUPLEX is not set (set by camilladsp_bufer_coreaudio.py).
const DUPLEX_BY_DEFAULT: bool = true;

/// Carries a value built on the control thread over to the IO callback, or back to be dropped.
/// The pipeline types have no thread affinity (plain data and `Arc`s), they only lack the bound.
struct Handoff<T>(T);
unsafe impl<T> Send for Handoff<T> {}

/// A parameter update on its way to the pipeline that lives in the callback.
type Update = (config::Configuration, Vec<String>, Vec<String>, ConvCoeffCache);

/// What the callback counts, for the log and for the measured sample rate.
#[derive(Default)]
struct Counters {
    callbacks: AtomicU64,
    max_callback_us: AtomicU64,
    max_gap_us: AtomicU64,
    input_errors: AtomicU64,
    skips: AtomicU64,
    skipped_frames: AtomicU64,
    wrong_size: AtomicU64,
    last_wrong_size: AtomicU64,
    stop: AtomicBool,
}

/// Whether this config asks for duplex mode and can run in it.
pub fn wanted(conf: &config::Configuration) -> bool {
    let asked = match std::env::var("CAMILLADSP_DUPLEX").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => DUPLEX_BY_DEFAULT,
    };
    if !asked {
        return false;
    }
    let devices = &conf.devices;
    let (config::CaptureDevice::CoreAudio(capture), config::PlaybackDevice::CoreAudio(playback)) =
        (&devices.capture, &devices.playback)
    else {
        return false;
    };
    let reason = if capture.device.is_none() || capture.device != playback.device {
        Some("capture and playback are not the same named device")
    } else if devices.resampler.is_some() || devices.capture_samplerate() != devices.samplerate() {
        Some("resampling is on")
    } else if devices.rate_adjust() {
        Some("rate adjust is on")
    } else if capture.format.is_some() || playback.format.is_some() {
        Some("a physical sample format is set")
    } else if playback.is_exclusive() {
        Some("exclusive mode is on")
    } else if devices.multithreaded() {
        Some("multithreaded processing is on")
    } else {
        None
    };
    match reason {
        Some(reason) => {
            info!("FJC duplex: not with this config ({reason}), using the regular threads.");
            false
        }
        None => true,
    }
}

/// Ask the device for `wanted` frames per IO buffer and return what it grants.
fn io_buffer_frames(device_id: AudioDeviceID, wanted: u32) -> u32 {
    let address = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyBufferFrameSize,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };
    let size = std::mem::size_of::<u32>() as u32;
    let mut actual: u32 = 0;
    unsafe {
        AudioObjectSetPropertyData(
            device_id,
            NonNull::from(&address),
            0,
            null(),
            size,
            NonNull::from(&wanted).cast(),
        );
        AudioObjectGetPropertyData(
            device_id,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&size),
            NonNull::from(&mut actual).cast(),
        );
    }
    actual
}

/// One audio unit on `device_id` with both its input and its output enabled.
fn open_duplex(
    device_id: AudioDeviceID,
    samplerate: usize,
    in_channels: usize,
    out_channels: usize,
) -> Result<AudioUnit, String> {
    let text = |e: coreaudio::Error| e.to_string();
    let mut audio_unit = AudioUnit::new(IOType::HalOutput).map_err(text)?;
    // The IO sides and the device are set on an uninitialized unit.
    audio_unit.uninitialize().map_err(text)?;
    let enable = 1u32;
    audio_unit
        .set_property(kAudioOutputUnitProperty_EnableIO, Scope::Input, Element::Input, Some(&enable))
        .map_err(text)?;
    audio_unit
        .set_property(kAudioOutputUnitProperty_EnableIO, Scope::Output, Element::Output, Some(&enable))
        .map_err(text)?;
    audio_unit
        .set_property(
            kAudioOutputUnitProperty_CurrentDevice,
            Scope::Global,
            Element::Output,
            Some(&device_id),
        )
        .map_err(text)?;
    set_device_sample_rate(device_id, samplerate as f64).map_err(text)?;
    let format = |channels: usize| {
        StreamFormat {
            sample_rate: samplerate as f64,
            sample_format: SampleFormat::F32,
            flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED,
            channels: channels as u32,
        }
        .to_asbd()
    };
    // What the callback writes, and what it reads from the input side.
    audio_unit
        .set_property(
            kAudioUnitProperty_StreamFormat,
            Scope::Input,
            Element::Output,
            Some(&format(out_channels)),
        )
        .map_err(text)?;
    audio_unit
        .set_property(
            kAudioUnitProperty_StreamFormat,
            Scope::Output,
            Element::Input,
            Some(&format(in_channels)),
        )
        .map_err(text)?;
    audio_unit.initialize().map_err(text)?;
    Ok(audio_unit)
}

/// Spawn the duplex thread. It opens the device, hands the pipeline to the IO callback and then
/// only waits for commands and config changes.
#[allow(clippy::too_many_arguments)]
pub fn start(
    conf: config::Configuration,
    barrier: Arc<Barrier>,
    status_channel: Sender<StatusMessage>,
    command_channel: Receiver<CommandMessage>,
    rx_pipeconf: Receiver<PipelineConfig>,
    capture_status: Arc<RwLock<CaptureStatus>>,
    playback_status: Arc<RwLock<PlaybackStatus>>,
    processing_params: Arc<ProcessingParameters>,
) -> Box<thread::JoinHandle<()>> {
    Box::new(
        thread::Builder::new()
            .name("CoreaudioDuplex".to_string())
            .spawn(move || {
                run(
                    conf,
                    barrier,
                    status_channel,
                    command_channel,
                    rx_pipeconf,
                    capture_status,
                    playback_status,
                    processing_params,
                )
            })
            .expect("can spawn duplex thread"),
    )
}

#[allow(clippy::too_many_arguments)]
fn run(
    conf: config::Configuration,
    barrier: Arc<Barrier>,
    status_channel: Sender<StatusMessage>,
    command_channel: Receiver<CommandMessage>,
    rx_pipeconf: Receiver<PipelineConfig>,
    capture_status: Arc<RwLock<CaptureStatus>>,
    playback_status: Arc<RwLock<PlaybackStatus>>,
    processing_params: Arc<ProcessingParameters>,
) {
    let samplerate = conf.devices.samplerate();
    let chunksize = conf.devices.chunksize();
    let (in_channels, out_channels, devname) = match (&conf.devices.capture, &conf.devices.playback)
    {
        (config::CaptureDevice::CoreAudio(capture), config::PlaybackDevice::CoreAudio(playback)) => (
            capture.channels.get(),
            playback.channels.get(),
            capture.device.clone().unwrap_or_default(),
        ),
        _ => unreachable!("duplex is only started for two CoreAudio devices"),
    };
    // An error before the start: report it and meet the supervisor at the barrier, like the
    // device threads do.
    let fail = |message: String| {
        error!("FJC duplex: {message}");
        status_channel
            .send(StatusMessage::PlaybackError(message))
            .unwrap_or(());
        barrier.wait();
    };

    processing_params.sync_volumes_to_target();
    let mut pipeline = Pipeline::from_config(
        conf.clone(),
        processing_params.clone(),
        None,
        &mut ConvCoeffCache::new(),
    );

    let device_id = match (
        get_device_id_from_name_and_scope(&devname, true),
        get_device_id_from_name_and_scope(&devname, false),
    ) {
        (Some(input), Some(output)) if input == output => input,
        _ => {
            return fail(format!(
                "'{devname}' is not one device with both inputs and outputs"
            ));
        }
    };
    let mut audio_unit = match open_duplex(device_id, samplerate, in_channels, out_channels) {
        Ok(audio_unit) => audio_unit,
        Err(err) => return fail(format!("could not open '{devname}': {err}")),
    };
    let granted = io_buffer_frames(device_id, chunksize as u32) as usize;
    if granted != chunksize {
        return fail(format!(
            "the device grants {granted} frames per buffer, set chunksize to that (it is {chunksize})"
        ));
    }
    info!(
        "FJC duplex: capture, processing and playback in the IO callback of '{devname}', {chunksize} frames per buffer, {in_channels} channels in and {out_channels} out."
    );

    // What the control thread and the callback pass each other. The callback only tries, it
    // never waits.
    let (tx_new, rx_new) = bounded::<Handoff<Pipeline>>(2);
    let (tx_update, rx_update) = bounded::<Handoff<Update>>(8);
    let (tx_old, rx_old) = bounded::<Handoff<Pipeline>>(4);
    let counters = Arc::new(Counters::default());

    {
        let counters = counters.clone();
        let capture_status = capture_status.clone();
        let playback_status = playback_status.clone();
        let used_channels = capture_status.read().used_channels.clone();
        let clipped_counter = playback_status.read().clipped_samples.clone();
        let instance = *audio_unit.as_ref();
        let mut in_buf = vec![0u8; 4 * in_channels * chunksize];
        let mut channel_mask = Vec::with_capacity(in_channels);
        let mut stats_in = ChunkStats {
            rms: vec![0.0; in_channels],
            peak: vec![0.0; in_channels],
        };
        let mut stats_out = ChunkStats {
            rms: vec![0.0; out_channels],
            peak: vec![0.0; out_channels],
        };
        let (mut rms_in, mut peak_in) = (Vec::with_capacity(in_channels), Vec::with_capacity(in_channels));
        let (mut rms_out, mut peak_out) = (Vec::with_capacity(out_channels), Vec::with_capacity(out_channels));
        let mut fjc_tp = crate::fjc_medidas::TruePeak::new(out_channels, chunksize);
        let mut running = false;
        let mut previous: Option<Instant> = None;
        let mut previous_sample_time: Option<f64> = None;
        // The device timeline can jump once right after the start, which is not a dropout.
        let warmup = (samplerate / chunksize / 2) as u64;
        let mut seen = 0u64;

        type Args = render_callback::Args<data::InterleavedBytes<f32>>;
        let callback = move |args: Args| {
            let Args {
                num_frames,
                data,
                time_stamp,
                ..
            } = args;
            let started = Instant::now();
            if let Some(previous) = previous {
                counters
                    .max_gap_us
                    .fetch_max((started - previous).as_micros() as u64, Ordering::Relaxed);
            }
            previous = Some(started);
            counters.callbacks.fetch_add(1, Ordering::Relaxed);
            // The device counts its frames: a jump means it skipped IO cycles (an overload).
            seen += 1;
            if let Some(previous) = previous_sample_time {
                let lost = time_stamp.mSampleTime - previous - num_frames as f64;
                if lost >= 1.0 && seen > warmup {
                    counters.skips.fetch_add(1, Ordering::Relaxed);
                    counters.skipped_frames.fetch_add(lost as u64, Ordering::Relaxed);
                }
            }
            previous_sample_time = Some(time_stamp.mSampleTime);
            let out = data.buffer;

            // A rebuilt pipeline replaces the running one, which goes back to be dropped elsewhere.
            if let Ok(Handoff(new_pipeline)) = rx_new.try_recv() {
                let old_pipeline = std::mem::replace(&mut pipeline, new_pipeline);
                let _ = tx_old.try_send(Handoff(old_pipeline));
            }
            while let Ok(Handoff((new_conf, filters, processors, mut coeff_cache))) = rx_update.try_recv() {
                pipeline.update_parameters(new_conf, &filters, &processors, &mut coeff_cache);
            }

            if counters.stop.load(Ordering::Relaxed) {
                out.fill(0);
                return Ok(());
            }
            if num_frames != chunksize {
                counters.wrong_size.fetch_add(1, Ordering::Relaxed);
                counters.last_wrong_size.store(num_frames as u64, Ordering::Relaxed);
                out.fill(0);
                return Ok(());
            }

            // The input of this same IO cycle, pulled from the input side of the unit.
            let needed = 4 * in_channels * num_frames;
            let mut buffer_list = AudioBufferList {
                mNumberBuffers: 1,
                mBuffers: [AudioBuffer {
                    mNumberChannels: in_channels as u32,
                    mDataByteSize: needed as u32,
                    mData: in_buf.as_mut_ptr() as *mut c_void,
                }],
            };
            let mut flags = AudioUnitRenderActionFlags(0);
            let status = unsafe {
                AudioUnitRender(
                    instance,
                    &mut flags,
                    NonNull::from(&time_stamp),
                    1,
                    num_frames as u32,
                    NonNull::from(&mut buffer_list),
                )
            };
            if status != 0 || buffer_list.mBuffers[0].mDataByteSize as usize != needed {
                counters.input_errors.fetch_add(1, Ordering::Relaxed);
                if running {
                    running = false;
                    warn!("Playback interrupted, no data available.");
                    debug!("FJC duplex: no input for this cycle, status {status}.");
                }
                out.fill(0);
                return Ok(());
            }
            if !running {
                running = true;
                debug!("FJC duplex: input is flowing.");
            }

            used_channels.copy_to(&mut channel_mask);
            let mut chunk = buffer_to_chunk_rawbytes(
                &in_buf[..needed],
                in_channels,
                &BinarySampleFormat::F32_LE,
                needed,
                &channel_mask,
                false,
            );
            chunk.update_stats(&mut stats_in);
            crate::push_capture_audio_buffer(&capture_status, &chunk);
            crate::update_capture_signal_status(&capture_status, &stats_in, &mut rms_in, &mut peak_in);

            chunk = pipeline.process_chunk(chunk);

            chunk.update_stats(&mut stats_out);
            fjc_tp.update(&chunk, &mut stats_out);
            crate::push_playback_audio_buffer(&playback_status, &chunk);
            let (_, clipped) = chunk_to_buffer_rawbytes(chunk, out, &BinarySampleFormat::F32_LE);
            crate::update_playback_signal_status(
                &playback_status,
                &stats_out,
                &mut rms_out,
                &mut peak_out,
                clipped,
                &clipped_counter,
            );
            counters
                .max_callback_us
                .fetch_max(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            Ok(())
        };
        if let Err(err) = audio_unit.set_render_callback(callback) {
            return fail(format!("could not set the callback: {err}"));
        }
    }

    let mut alive_listener = AliveListener::new(device_id);
    if let Err(err) = alive_listener.register() {
        warn!("FJC duplex: unable to register the device alive listener, error: {err}.");
    }
    let (rate_tx, rate_rx) = mpsc::channel();
    let mut rate_listener = RateListener::new(device_id, Some(rate_tx));
    if let Err(err) = rate_listener.register() {
        warn!("FJC duplex: unable to register the rate listener, error: {err}.");
    }

    status_channel.send(StatusMessage::PlaybackReady).unwrap_or(());
    status_channel.send(StatusMessage::CaptureReady).unwrap_or(());
    barrier.wait();
    debug!("FJC duplex: starts now!");
    // After an error the thread stays, silent, until the supervisor tells it to exit.
    let mut failed = false;
    if let Err(err) = audio_unit.start() {
        status_channel
            .send(StatusMessage::PlaybackError(err.to_string()))
            .unwrap_or(());
        failed = true;
    } else {
        let mut capture_status = capture_status.write();
        capture_status.measured_samplerate = samplerate;
        crate::update_capture_state(&mut capture_status, ProcessingState::Running);
    }

    let log_stats = std::env::var("CAMILLADSP_FILL_STATS").is_ok();
    let mut stats_timer = Instant::now();
    let mut rate_timer = Instant::now();
    let mut rate_callbacks = 0u64;
    loop {
        match command_channel.recv_timeout(Duration::from_millis(20)) {
            Ok(CommandMessage::Exit) => {
                debug!("FJC duplex: exit message received.");
                break;
            }
            Ok(CommandMessage::SetSpeed { .. }) => {}
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                error!("FJC duplex: command channel was closed.");
                break;
            }
        }
        // Config changes: a rebuild is done here and swapped in, parameters go to the callback.
        while let Ok((change, new_conf, mut coeff_cache)) = rx_pipeconf.try_recv() {
            match change {
                config::ConfigChange::Pipeline | config::ConfigChange::MixerParameters => {
                    info!("FJC duplex: rebuilding the pipeline.");
                    processing_params.sync_volumes_to_target();
                    let new_pipeline = Pipeline::from_config(
                        new_conf,
                        processing_params.clone(),
                        None,
                        &mut coeff_cache,
                    );
                    tx_new.send(Handoff(new_pipeline)).unwrap_or(());
                }
                config::ConfigChange::FilterParameters {
                    filters,
                    processors,
                } => {
                    info!("FJC duplex: updating parameters of filters: {filters:?}.");
                    tx_update.send(Handoff((new_conf, filters, processors, coeff_cache))).unwrap_or(());
                }
                config::ConfigChange::Devices | config::ConfigChange::None => {}
            }
        }
        while rx_old.try_recv().is_ok() {}

        if failed {
            continue;
        }
        if !alive_listener.is_alive() {
            error!("FJC duplex: the device is no longer alive.");
            counters.stop.store(true, Ordering::Relaxed);
            status_channel
                .send(StatusMessage::PlaybackError(
                    "Device is no longer alive".to_string(),
                ))
                .unwrap_or(());
            failed = true;
            continue;
        }
        if let Ok(rate) = rate_rx.try_recv() {
            debug!("FJC duplex: rate change event, new rate: {rate}.");
            if rate as usize != samplerate {
                counters.stop.store(true, Ordering::Relaxed);
                status_channel
                    .send(StatusMessage::PlaybackFormatChange(rate as usize))
                    .unwrap_or(());
                failed = true;
                continue;
            }
        }
        let skips = counters.skips.swap(0, Ordering::Relaxed);
        if skips > 0 {
            let frames = counters.skipped_frames.swap(0, Ordering::Relaxed);
            warn!("Playback interrupted, the device skipped {frames} frames in {skips} IO cycles.");
        }
        if counters.wrong_size.swap(0, Ordering::Relaxed) > 0 {
            let frames = counters.last_wrong_size.load(Ordering::Relaxed);
            counters.stop.store(true, Ordering::Relaxed);
            status_channel
                .send(StatusMessage::PlaybackError(format!(
                    "The device now asks for {frames} frames per buffer, chunksize is {chunksize}"
                )))
                .unwrap_or(());
            failed = true;
            continue;
        }
        // The sample rate, measured from the callbacks, for the status.
        if rate_timer.elapsed() >= Duration::from_secs(1) {
            let callbacks = counters.callbacks.load(Ordering::Relaxed);
            let measured = (callbacks - rate_callbacks) as f64 * chunksize as f64
                / rate_timer.elapsed().as_secs_f64();
            rate_callbacks = callbacks;
            rate_timer = Instant::now();
            if let Some(mut capture_status) = capture_status.try_write() {
                capture_status.measured_samplerate = measured.round() as usize;
            }
        }
        if log_stats && stats_timer.elapsed() >= Duration::from_secs(5) {
            stats_timer = Instant::now();
            info!(
                "FJC duplex stat: callbacks {}; max callback us {}; max gap us {}; input errors {}.",
                counters.callbacks.load(Ordering::Relaxed),
                counters.max_callback_us.swap(0, Ordering::Relaxed),
                counters.max_gap_us.swap(0, Ordering::Relaxed),
                counters.input_errors.swap(0, Ordering::Relaxed),
            );
        }
    }
    if let Err(err) = audio_unit.stop() {
        warn!("FJC duplex: error stopping the audio unit: {err}.");
    }
    // Dropping the unit frees the callback, and with it the pipeline, on this thread.
    drop(audio_unit);
    debug!("FJC duplex: stopped.");
}
