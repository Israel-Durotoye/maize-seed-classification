import "./style.css";

type SeedClass = "BAD_SEED" | "GOOD_SEED" | "NO_MAIZE";

interface SeedEvent {
  id: number;
  time: number;
  label: "GOOD_SEED" | "BAD_SEED" | "UNSURE";
  reason: "classified" | "uncertain_reject";
  confidence: number | null;
}

interface State {
  status: "starting" | "waiting" | "armed" | "error";
  message: string;
  armed: boolean;
  camera: string;
  fps: number;
  inference_ms: number;
  counts: { good: number; bad: number; uncertain: number; total: number };
  prediction: null | {
    label: SeedClass;
    confidence: number;
    margin: number;
    probabilities: Record<SeedClass, number>;
    eligible: boolean;
    empty: boolean;
  };
  events: SeedEvent[];
}

interface ModelInfo {
  run: string;
  architecture: string;
  test_accuracy: number | null;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  status: $("status"),
  statusText: $("statusText"),
  feed: $<HTMLImageElement>("feed"),
  feedMessage: $("feedMessage"),
  chip: $("liveChip"),
  liveLabel: $("liveLabel"),
  liveConf: $("liveConf"),
  flash: $("flash"),
  hint: $("hint"),
  fps: $("fps"),
  latency: $("latency"),
  cameraSelect: $<HTMLSelectElement>("cameraSelect"),
  good: $("goodCount"),
  bad: $("badCount"),
  unsure: $("unsureCount"),
  total: $("totalCount"),
  ratio: document.querySelector(".ratio") as HTMLElement,
  ratioGood: $("ratioGood"),
  goodPct: $("goodPct"),
  badPct: $("badPct"),
  events: $<HTMLOListElement>("events"),
  reset: $<HTMLButtonElement>("resetBtn"),
  modelInfo: $("modelInfo"),
};

const LABELS: Record<SeedClass, string> = { GOOD_SEED: "Good seed", BAD_SEED: "Bad seed", NO_MAIZE: "No seed" };
const STATUS_TEXT: Record<State["status"] | "offline" | "connecting", string> = {
  connecting: "Connecting…",
  starting: "Starting camera",
  waiting: "Waiting for empty view",
  armed: "Live · ready",
  error: "Camera problem",
  offline: "Server offline",
};

const pct = (x: number) => `${Math.round(x * 100)}%`;

let last: State | null = null;
let lastEventId = 0;

function setStatus(state: keyof typeof STATUS_TEXT) {
  els.status.dataset.state = state;
  els.statusText.textContent = STATUS_TEXT[state];
}

function bumpCounter(el: HTMLElement, kind: "good" | "bad", added: number) {
  el.classList.remove("bump");
  void el.offsetWidth; // Restart the animation.
  el.classList.add("bump");
  const plus = document.createElement("span");
  plus.className = "plus";
  plus.textContent = `+${added}`;
  el.parentElement!.appendChild(plus);
  plus.addEventListener("animationend", () => plus.remove());

  els.flash.className = `flash ${kind}`;
  requestAnimationFrame(() => setTimeout(() => (els.flash.className = "flash"), 250));
}

function renderCounts(s: State) {
  const { good, bad, uncertain, total } = s.counts;
  if (last) {
    if (good > last.counts.good) bumpCounter(els.good, "good", good - last.counts.good);
    if (bad > last.counts.bad) bumpCounter(els.bad, "bad", bad - last.counts.bad);
  }
  els.good.textContent = String(good);
  els.bad.textContent = String(bad);
  els.total.textContent = String(total);
  els.unsure.textContent = uncertain ? `· ${uncertain} unsure, not counted` : "";

  els.ratio.classList.toggle("empty", total === 0);
  const goodShare = total ? good / total : 0;
  els.ratioGood.style.width = `${goodShare * 100}%`;
  els.goodPct.textContent = total ? pct(goodShare) : "0%";
  els.badPct.textContent = total ? pct(1 - goodShare) : "0%";
}

function renderPrediction(s: State) {
  const p = s.prediction;
  for (const row of document.querySelectorAll<HTMLElement>(".prob-row")) {
    const value = p ? p.probabilities[row.dataset.cls as SeedClass] : 0;
    (row.querySelector(".bar > div") as HTMLElement).style.width = pct(value);
    row.querySelector("em")!.textContent = pct(value);
  }
  const live = s.status === "armed" || s.status === "waiting";
  els.chip.hidden = !p || !live;
  if (p) {
    els.chip.dataset.cls = p.label;
    els.liveLabel.textContent = LABELS[p.label];
    els.liveConf.textContent = pct(p.confidence);
  }
}

function renderEvents(s: State) {
  const topId = s.events[0]?.id ?? 0;
  if (topId === lastEventId && s.events.length === els.events.querySelectorAll("li:not(.empty)").length) return;
  els.events.replaceChildren();
  if (!s.events.length) {
    els.events.innerHTML = '<li class="empty">No seeds counted yet.</li>';
  }
  for (const e of s.events.slice(0, 12)) {
    const li = document.createElement("li");
    if (e.id > lastEventId && lastEventId !== 0) li.classList.add("fresh");
    const time = new Date(e.time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const tag = { GOOD_SEED: "Good", BAD_SEED: "Bad", UNSURE: "Unsure" }[e.label];
    const detail = e.confidence != null ? pct(e.confidence) : "low confidence";
    li.innerHTML = `<span class="num">#${e.id}</span><span class="tag ${e.label}">${tag}</span>
      <span class="detail" title="${time}">${detail}</span>`;
    els.events.appendChild(li);
  }
  lastEventId = topId;
}

function render(s: State) {
  setStatus(s.status);
  els.hint.textContent = s.message;
  els.fps.textContent = `${s.fps.toFixed(1)} fps`;
  els.latency.textContent = `${Math.round(s.inference_ms)} ms`;
  const offline = s.status === "error" || s.status === "starting";
  els.feedMessage.hidden = !offline;
  if (offline) els.feedMessage.textContent = s.message;
  renderCounts(s);
  renderPrediction(s);
  renderEvents(s);
  last = s;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => render(JSON.parse(msg.data) as State);
  ws.onclose = () => {
    setStatus("offline");
    els.chip.hidden = true;
    els.feedMessage.hidden = false;
    els.feedMessage.textContent = "Can't reach the sorter server. Is web/backend/server.py running?";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

function startFeed() {
  els.feed.src = `/api/stream?t=${Date.now()}`;
}
els.feed.onerror = () => setTimeout(startFeed, 2000);

async function loadCameras() {
  try {
    const res = await fetch("/api/cameras");
    const { current, available } = (await res.json()) as { current: string; available: number[] };
    const sources = new Set<string>([...available.map(String), current]);
    els.cameraSelect.replaceChildren(
      ...[...sources].map((src) => new Option(/^\d+$/.test(src) ? `Camera ${src}` : src, src, false, src === current)),
    );
  } catch {
    setTimeout(loadCameras, 3000);
  }
}

els.cameraSelect.addEventListener("change", async () => {
  await fetch("/api/camera", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: els.cameraSelect.value }),
  });
});

els.reset.addEventListener("click", async () => {
  if (!confirm("Reset the good and bad seed counts to zero?")) return;
  last = null; // Don't animate the drop to zero.
  lastEventId = 0;
  const res = await fetch("/api/reset", { method: "POST" });
  render((await res.json()) as State);
});

async function loadModel() {
  try {
    const info = (await (await fetch("/api/model")).json()) as ModelInfo;
    const acc = info.test_accuracy != null ? ` · ${(info.test_accuracy * 100).toFixed(1)}% test accuracy` : "";
    els.modelInfo.textContent = `${info.architecture} · run ${info.run}${acc}`;
  } catch {
    setTimeout(loadModel, 3000);
  }
}

connect();
startFeed();
loadCameras();
loadModel();
