// Turns per-frame detections into the few sentences worth saying aloud.
// Pure logic with an injected clock, so Node can test it.
//
// Rules, in the order they apply:
//   1. A thing must be seen in consecutive frames before it is spoken.
//   2. The same alert is not repeated within the cooldown, unless the
//      object has come close or moved to another direction since.
//   3. One sentence at a time; the most urgent wins. Only a close vehicle
//      may interrupt speech already in progress.
//   4. Silence is the default. A system that talks constantly gets muted.

import { ALERT_TIMING } from './config.js';

export const ZONES = ['left', 'ahead', 'right'];
const ZONE_PHRASE = { left: 'on your left', ahead: 'ahead', right: 'on your right' };

export function zoneOf(det) {
  const cx = (det.x1 + det.x2) / 2;
  if (cx < 0.36) return 'left';
  if (cx > 0.64) return 'right';
  return 'ahead';
}

/**
 * Rough nearness from the box alone, assuming the phone is held upright
 * and facing forward. Things on the ground are close when their lower edge
 * nears the bottom of the frame; things standing up are close when they
 * fill much of its height. A heuristic, not a distance measurement.
 */
export function isClose(det, kind) {
  if (kind === 'ground') return det.y2 > 0.82;
  return det.y2 - det.y1 > 0.55 || (det.y2 > 0.9 && det.y2 - det.y1 > 0.35);
}

export function phrase(word, zone, close) {
  const w = word.charAt(0).toUpperCase() + word.slice(1);
  return `${w} ${ZONE_PHRASE[zone]}${close ? ', close' : ''}`;
}

export class AlertEngine {
  constructor(classes, timing = ALERT_TIMING) {
    this.classes = classes;
    this.timing = timing;
    this.tracks = new Map(); // spoken word -> state
  }

  reset() {
    this.tracks.clear();
  }

  /**
   * @param dets        detections in source coordinates ({cls, score, x1..y2})
   * @param nowMs       monotonic clock
   * @param opts.fps            current processing rate
   * @param opts.speechBusy     true while a sentence is being spoken
   * @param opts.announceFurniture  include priority-4 classes
   * @returns {speak: {text, priority, interrupt, word, zone, close} | null,
   *           groups: [...]}  groups are what is currently confirmed in view
   */
  update(dets, nowMs, opts = {}) {
    const fps = opts.fps || 0;
    // At low frame rates three frames is a long time to wait.
    const needFrames = fps > 0 && fps < 5 ? 2 : 3;

    // 1. Group this frame's detections by the word that would be spoken.
    const groups = new Map();
    for (const d of dets) {
      const c = this.classes[d.cls];
      if (!c || !c.announce) continue;
      if (c.priority >= 4 && !opts.announceFurniture) continue;
      const close = isClose(d, c.kind);
      const size = (d.x2 - d.x1) * (d.y2 - d.y1);
      let g = groups.get(c.say);
      if (!g) {
        g = { word: c.say, plural: c.plural, priority: c.priority, count: 0, lead: null, leadRank: -1 };
        groups.set(c.say, g);
      }
      g.count++;
      // The detection that decides direction: a close one first, then the biggest.
      const rank = (close ? 10 : 0) + size;
      if (rank > g.leadRank) {
        g.leadRank = rank;
        g.lead = d;
        g.zone = zoneOf(d);
        g.close = close;
        g.score = d.score;
      }
    }

    // 2. Update what has been seen for how long. One missed frame is
    //    forgiven; a longer gap starts the count again.
    const gapMs = fps > 0 ? Math.max(700, 2500 / fps) : 700;
    for (const word of groups.keys()) {
      let t = this.tracks.get(word);
      if (!t) {
        t = { hits: 0, lastSeen: nowMs, spokenAt: -Infinity, spokenZone: null, spokenClose: false };
        this.tracks.set(word, t);
      }
      if (nowMs - t.lastSeen > gapMs) t.hits = 0;
      t.hits += 1;
      t.lastSeen = nowMs;
    }
    for (const [word, t] of this.tracks) {
      if (groups.has(word)) continue;
      if (nowMs - t.lastSeen > this.timing.forgetMs) {
        // Gone. Keep spokenAt so a flickering detection cannot re-announce
        // itself, but let "close" be reported again next time.
        t.hits = 0;
        t.spokenClose = false;
        if (nowMs - t.spokenAt > 4 * this.timing.cooldownMs) this.tracks.delete(word);
      }
    }

    // 3. Which confirmed group, if any, deserves a sentence now?
    const confirmed = [];
    let pick = null;
    for (const [word, g] of groups) {
      const t = this.tracks.get(word);
      if (t.hits < needFrames) continue;
      confirmed.push(g);
      const since = nowMs - t.spokenAt;
      const due =
        since >= this.timing.cooldownMs ||
        (g.close && !t.spokenClose && since >= this.timing.escalateMs) ||
        (g.zone !== t.spokenZone && since >= this.timing.zoneChangeMs);
      if (!due) continue;
      if (
        !pick ||
        g.priority < pick.priority ||
        (g.priority === pick.priority && g.close && !pick.close) ||
        (g.priority === pick.priority && g.close === pick.close && g.score > pick.score)
      ) {
        pick = g;
      }
    }

    let speak = null;
    if (pick) {
      const urgent = pick.priority === 1 && pick.close;
      if (!opts.speechBusy || urgent) {
        const t = this.tracks.get(pick.word);
        t.spokenAt = nowMs;
        t.spokenZone = pick.zone;
        t.spokenClose = pick.close;
        speak = {
          text: phrase(pick.count > 1 ? pick.plural : pick.word, pick.zone, pick.close),
          priority: pick.priority,
          interrupt: urgent && !!opts.speechBusy,
          word: pick.word,
          zone: pick.zone,
          close: pick.close,
        };
      }
    }
    return { speak, groups: confirmed };
  }
}
