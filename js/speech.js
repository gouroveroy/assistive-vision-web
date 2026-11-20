// Thin wrapper over the browser's speech synthesis.
//
// iOS Safari only lets a page speak after a tap, so unlock() must be called
// from the Start button's click handler. After that, speak() works from
// anywhere.

export class Speaker {
  constructor() {
    this.synth = typeof window !== 'undefined' ? window.speechSynthesis : null;
    this.rate = 1.1;
    this.voice = null;
    this.lastStart = 0;
    this.log = []; // what was said, newest last (also used by the tests)
    if (this.synth) {
      const pick = () => {
        const voices = this.synth.getVoices();
        this.voice =
          voices.find((v) => v.lang === 'en-US' && v.localService) ||
          voices.find((v) => v.lang && v.lang.startsWith('en') && v.localService) ||
          voices.find((v) => v.lang && v.lang.startsWith('en')) ||
          null;
      };
      pick();
      if (typeof this.synth.addEventListener === 'function') {
        this.synth.addEventListener('voiceschanged', pick);
      }
    }
  }

  get available() {
    return !!this.synth;
  }

  /** True while a sentence is being spoken or waiting to be. */
  get busy() {
    if (!this.synth) return false;
    if (!(this.synth.speaking || this.synth.pending)) return false;
    // Some browsers leave `speaking` stuck after an interrupted utterance.
    return performance.now() - this.lastStart < 6000;
  }

  /** Call from a user gesture. Speaks a short line, which unlocks iOS. */
  unlock(text) {
    this.speak(text, { interrupt: true });
  }

  speak(text, { interrupt = false } = {}) {
    this.log.push({ text, at: Date.now() });
    if (this.log.length > 50) this.log.shift();
    if (!this.synth) return;
    try {
      if (interrupt) this.synth.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'en-US';
      u.rate = this.rate;
      if (this.voice) u.voice = this.voice;
      this.lastStart = performance.now();
      this.synth.speak(u);
    } catch (err) {
      console.warn('speech failed', err);
    }
  }

  stop() {
    if (this.synth) this.synth.cancel();
  }
}
