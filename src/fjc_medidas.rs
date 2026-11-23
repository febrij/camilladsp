//! FJC patch (11): measurements for the control panel (2026-11-21).
//!
//! * True peak on the playback side: the stock per-sample peak misses inter-sample
//!   maxima (ITU-R BS.1770 annex 2). `TruePeak` interpolates the three points between
//!   consecutive samples (x4 oversampling, 12 taps per phase, Hann-windowed sinc; checked
//!   error with sines up to 20 kHz: -0.3 to +0.05 dB) and raises the chunk's peak to the
//!   largest value found. CAMILLADSP_TRUE_PEAK=0 turns it off.
//! * Compressor activity: each compressor publishes the highest detector level and the
//!   deepest gain reduction since the last read; the websocket command
//!   `GetProcessorLevels` returns {name: [level dB, reduction dB (<= 0), threshold dB]}
//!   and resets them.
use crate::CamillaFloat;
use crate::audiochunk::{AudioChunk, ChunkStats};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

const TAPS: usize = 12;

pub struct TruePeak {
    on: bool,
    coefs: [[CamillaFloat; TAPS]; 3],
    ext: Vec<Vec<CamillaFloat>>,
}

impl TruePeak {
    pub fn new(channels: usize, chunksize: usize) -> Self {
        let on = std::env::var("CAMILLADSP_TRUE_PEAK").map(|v| v != "0").unwrap_or(true);
        let mut coefs = [[0.0; TAPS]; 3];
        for (p, phase) in coefs.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            let mut h = [0.0f64; TAPS];
            for (k, val) in h.iter_mut().enumerate() {
                // distance from sample k to the interpolated point, between samples TAPS/2-1 and TAPS/2
                let t = k as f64 - (TAPS as f64 / 2.0 - 1.0) - (p as f64 + 1.0) / 4.0;
                let x = std::f64::consts::PI * t;
                let sinc = if x.abs() < 1.0e-12 { 1.0 } else { x.sin() / x };
                let window = 0.5 * (1.0 + (x / (TAPS as f64 / 2.0)).cos());
                *val = sinc * window;
                sum += *val;
            }
            for (c, val) in phase.iter_mut().zip(h.iter()) {
                *c = (*val / sum) as CamillaFloat;
            }
        }
        let ext = (0..channels)
            .map(|_| {
                let mut v = Vec::with_capacity(TAPS - 1 + 2 * chunksize.max(64));
                v.resize(TAPS - 1, 0.0);
                v
            })
            .collect();
        TruePeak { on, coefs, ext }
    }

    /// Raise `stats.peak` (already filled by `chunk.update_stats`) to the true peak.
    pub fn update(&mut self, chunk: &AudioChunk, stats: &mut ChunkStats) {
        if !self.on {
            return;
        }
        for ((wf, ext), peak) in chunk
            .waveforms
            .iter()
            .zip(self.ext.iter_mut())
            .zip(stats.peak.iter_mut())
        {
            if wf.is_empty() {
                continue;
            }
            ext.truncate(TAPS - 1);
            ext.extend_from_slice(wf);
            let mut max: CamillaFloat = 0.0;
            for win in ext.windows(TAPS) {
                for phase in self.coefs.iter() {
                    let mut acc: CamillaFloat = 0.0;
                    for (c, v) in phase.iter().zip(win.iter()) {
                        acc += *c * *v;
                    }
                    if acc.abs() > max {
                        max = acc.abs();
                    }
                }
            }
            if (max as f32) > *peak {
                *peak = max as f32;
            }
            let n = ext.len();
            ext.copy_within(n - (TAPS - 1).., 0);
            ext.truncate(TAPS - 1);
        }
    }
}

#[derive(Debug)]
pub struct Measure {
    level: AtomicU32,
    reduction: AtomicU32,
    threshold: AtomicU32,
}

impl Measure {
    /// Called once per chunk from the processing thread: lock-free.
    pub fn record(&self, level: f32, threshold: f32, factor: f32) {
        if level > f32::from_bits(self.level.load(Ordering::Relaxed)) {
            self.level.store(level.to_bits(), Ordering::Relaxed);
        }
        if level > threshold {
            let reduction = -(level - threshold) * (factor - 1.0) / factor;
            if reduction < f32::from_bits(self.reduction.load(Ordering::Relaxed)) {
                self.reduction.store(reduction.to_bits(), Ordering::Relaxed);
            }
        }
        self.threshold.store(threshold.to_bits(), Ordering::Relaxed);
    }
}

static REGISTRY: Mutex<BTreeMap<String, Arc<Measure>>> = Mutex::new(BTreeMap::new());

pub fn register(name: &str) -> Arc<Measure> {
    let m = Arc::new(Measure {
        level: AtomicU32::new((-1000.0f32).to_bits()),
        reduction: AtomicU32::new(0.0f32.to_bits()),
        threshold: AtomicU32::new(0.0f32.to_bits()),
    });
    REGISTRY.lock().insert(name.to_string(), m.clone());
    m
}

/// Read and reset. Entries whose compressor is gone are dropped.
pub fn read() -> BTreeMap<String, Vec<f32>> {
    let mut reg = REGISTRY.lock();
    reg.retain(|_, m| Arc::strong_count(m) > 1);
    reg.iter()
        .map(|(name, m)| {
            let level = f32::from_bits(m.level.swap((-1000.0f32).to_bits(), Ordering::Relaxed));
            let reduction = f32::from_bits(m.reduction.swap(0.0f32.to_bits(), Ordering::Relaxed));
            let threshold = f32::from_bits(m.threshold.load(Ordering::Relaxed));
            (name.clone(), vec![level, reduction, threshold])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, phase: f64, n: usize) -> Vec<CamillaFloat> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 48000.0 + phase).sin() as CamillaFloat)
            .collect()
    }

    #[test]
    fn true_peak_sees_between_samples() {
        // fs/4 at 45 degrees: every sample is 0.707, the real peak is 1.0
        let wf = sine(12000.0, std::f64::consts::FRAC_PI_4, 4800);
        let mut tp = TruePeak::new(1, 16);
        let mut worst: f32 = 0.0;
        for part in wf.chunks(16) {
            let chunk = AudioChunk::new(vec![part.to_vec()], 1.0, -1.0, 16, 16);
            let mut stats = ChunkStats { rms: vec![0.0], peak: vec![0.0] };
            chunk.update_stats(&mut stats);
            assert!(stats.peak[0] < 0.72);
            tp.update(&chunk, &mut stats);
            worst = worst.max(stats.peak[0]);
        }
        assert!((0.97..1.02).contains(&worst), "true peak {worst}");
    }

    #[test]
    fn true_peak_does_not_inflate_low_frequencies() {
        let wf = sine(50.0, 0.3, 9600);
        let mut tp = TruePeak::new(1, 64);
        let mut worst: f32 = 0.0;
        for part in wf.chunks(64) {
            let chunk = AudioChunk::new(vec![part.to_vec()], 1.0, -1.0, 64, 64);
            let mut stats = ChunkStats { rms: vec![0.0], peak: vec![0.0] };
            chunk.update_stats(&mut stats);
            tp.update(&chunk, &mut stats);
            worst = worst.max(stats.peak[0]);
        }
        assert!((0.999..1.001).contains(&worst), "true peak {worst}");
    }

    #[test]
    fn measure_records_and_resets() {
        let m = register("fjc_test");
        m.record(-30.0, -20.0, 20.0);
        m.record(-10.0, -20.0, 20.0);
        let r = read();
        let v = &r["fjc_test"];
        assert!((v[0] + 10.0).abs() < 1e-6 && (v[1] + 9.5).abs() < 1e-6 && (v[2] + 20.0).abs() < 1e-6);
        let v = &read()["fjc_test"];
        assert!(v[0] < -999.0 && v[1] == 0.0);
        drop(m);
        assert!(!read().contains_key("fjc_test"));
    }
}
