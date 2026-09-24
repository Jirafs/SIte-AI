import { CreateMLCEngine, prebuiltAppConfig } from "https://esm.run/@mlc-ai/web-llm@0.2.84";

/* ==================== МОДЕЛИ ==================== */

// Предпочтения по качеству (используются только те, что реально есть в web-llm).
const PREFERRED = [
  "Qwen2.5-7B-Instruct-q4f16_1-MLC",
  "Qwen2.5-3B-Instruct-q4f16_1-MLC",
  "Llama-3.2-3B-Instruct-q4f16_1-MLC",
  "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
  "Llama-3.2-1B-Instruct-q4f16_1-MLC",
];

// Человекочитаемые имена для select.
const NICE_NAMES = {
  "Qwen2.5-7B-Instruct-q4f16_1-MLC": "Qwen 2.5 7B — лучший ответ, нужен GPU ~6 ГБ",
  "Qwen2.5-3B-Instruct-q4f16_1-MLC": "Qwen 2.5 3B — баланс, GPU ~3 ГБ",
  "Llama-3.2-3B-Instruct-q4f16_1-MLC": "Llama 3.2 3B — альтернатива",
  "Qwen2.5-1.5B-Instruct-q4f16_1-MLC": "Qwen 2.5 1.5B — слабый ПК",
  "Llama-3.2-1B-Instruct-q4f16_1-MLC": "Llama 3.2 1B — самый быстрый",
};

function availableModels() {
  const list = (prebuiltAppConfig && prebuiltAppConfig.model_list) || [];
  const byId = new Map(list.map((m) => [m.model_id, m]));
  const preferred = PREFERRED.filter((id) => byId.has(id));
  if (preferred.length >= 2) return preferred;

  // Если предпочтительных почти нет — берём обычные LLM-модели по убыванию VRAM.
  return list
    .filter((m) => !m.model_type && m.vram_required_MB && m.vram_required_MB <= 6500)
    .sort((a, b) => b.vram_required_MB - a.vram_required_MB)
    .slice(0, 5)
    .map((m) => m.model_id);
}

function modelLabel(id) {
  if (NICE_NAMES[id]) return NICE_NAMES[id].split(" — ")[0];
  return String(id).replace(/-q4f16_1-MLC$/, "").replace(/-MLC$/, "");
}

/* ==================== ПОИСК ПО МЕТОДИЧКЕ (RAG) ==================== */

const STOPWORDS = new Set(
  `и в во на по с со к ко о об от до из у за для как что это а не ни же или то я мы вы он она они ты меня тебя ему ей нас вас их быть был была были будет буду будут есть был этот эта эти эти том чем какой какая какие какие-то про при над под без да нет уже ещё еще очень можно нужно надо чтобы чтоб вот так такой такая такие если то есть тд т.д № гг год года всего всего также между перед после около через многих многих весь вся всё все моя мой наш ваш свой себя себя сам сама`
    .split(/\s+/)
    .filter(Boolean)
);

// Группы синонимов: всё в группе сводится к первому слову.
const SYNONYMS = [
  ["эксель", "excel", "ексель", "мсэксель", "экселе", "экселю", "worksheet"],
  ["ворд", "word", "мсворд", "ворде", "документ.word"],
  ["пауэрпойнт", "powerpoint", "презентация", "презентации", "слайд", "слайды", "слайдов"],
  ["фигма", "figma", "figme", "фигме", "фигмой"],
  ["диаграмма", "диаграммы", "диаграмм", "график", "графики", "графиков", "гистограмма", "круговая"],
  ["формула", "формулы", "формул", "функция", "функции", "функций"],
  ["таблица", "таблицы", "таблиц", "таблице", "таблицах"],
  ["ссылка", "ссылки", "ссылок", "ссылке", "адресация", "абсолютная", "относительная"],
  ["нейросеть", "нейросети", "нейронные", "нейронных", "ии", "искусственный"],
  ["промпт", "промпты", "промптов", "запрос", "запросы", "запросов", "шаблон"],
  ["чеклист", "чек-лист", "чек-листа", "чек-листу", "критерии", "критериев"],
  ["дашборд", "dashboard", "панель"],
  ["макрос", "макросы", "vba", "макросов"],
  ["сортировка", "фильтр", "фильтрация", "фильтры", "проверкаданных"],
  ["сводная", "сводные", "сводных", "сводная"],
  ["форматирование", "оформление", "стиль", "стили", "стилей"],
  ["практическая", "практическое", "пз", "лабораторная", "задании", "занятие", "занятий"],
  ["фамилия", "фио", "студент", "студента"],
  ["leonardo", "spline", "pixverse", "генерация", "изображение", "изображений"],
  ["вычисление", "вычисления", "расчет", "расчёт", "расчеты", "расчёты", "kpi"],
];

const SYN_MAP = new Map();
for (const group of SYNONYMS) {
  for (const word of group) SYN_MAP.set(word.replace(/ё/g, "е"), group[0]);
}

const SUFFIXES = [
  "иями", "иях", "иям", "ией", "ием", "ия", "ами", "ями", "ого", "его", "ому", "ему",
  "ыми", "ими", "ов", "ев", "ам", "ям", "ах", "ях", "ие", "ые", "ая", "яя", "ое", "ее",
  "ой", "ей", "ый", "ий", "ый", "у", "ю", "а", "я", "о", "е", "ы", "и", "ь",
].sort((a, b) => b.length - a.length);

function stem(word) {
  for (const suf of SUFFIXES) {
    if (word.length - suf.length >= 3 && word.endsWith(suf)) return word.slice(0, -suf.length);
  }
  return word;
}

function tokenize(text) {
  const raw = String(text || "")
    .toLowerCase()
    .replace(/ё/g, "е")
    .match(/[a-zа-я0-9]{2,}/g);
  if (!raw) return [];
  const out = [];
  for (let w of raw) {
    if (w.length === 2 && /\d/.test(w)) continue;
    if (STOPWORDS.has(w)) continue;
    w = SYN_MAP.get(w) || w;
    out.push(stem(w));
  }
  return out;
}

const BM25_K1 = 1.4;
const BM25_B = 0.75;

class SearchIndex {
  constructor(chunks) {
    this.chunks = chunks;
    this.docTokens = [];
    this.docTf = [];
    this.titleTf = [];
    this.df = new Map();
    this.totalLen = 0;

    chunks.forEach((chunk, i) => {
      const tokens = tokenize(chunk.text);
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      this.docTokens.push(tokens);
      this.docTf.push(tf);
      this.titleTf.push(new Map(Array.from(new Set(tokenize(chunk.title))).map((t) => [t, 1])));
      this.totalLen += tokens.length;
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
    });

    this.avgLen = chunks.length ? this.totalLen / chunks.length : 1;
    this.n = chunks.length || 1;
  }

  idf(term) {
    const df = this.df.get(term) || 0;
    return Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
  }

  // Итоговый балл чанка: BM25 по тексту + усиление за совпадения в заголовке.
  scoreDoc(i, qTerms) {
    const tf = this.docTf[i];
    const len = this.docTokens[i].length || 1;
    let score = 0;
    for (const t of qTerms) {
      const f = tf.get(t);
      if (!f) continue;
      score += this.idf(t) * ((f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * (len / this.avgLen))));
    }
    let titleScore = 0;
    for (const t of qTerms) titleScore += (this.titleTf[i].get(t) ? this.idf(t) : 0) * 2.2;
    return score + titleScore;
  }

  // Дополнительный бонус за совпадение корней («формул» ↔ «формулировк»).
  prefixBonus(i, qTerms) {
    const stems = this.docTokens[i];
    let bonus = 0;
    for (const qt of qTerms) {
      if (qt.length < 5) continue;
      for (const dt of stems) {
        if (dt === qt) continue;
        if (dt.startsWith(qt) || qt.startsWith(dt)) {
          bonus += 0.35;
          break;
        }
      }
    }
    return bonus;
  }
}

/* ---------- сбор и нарезка текста методички ---------- */

const BLOCK_SELECTOR = "h2, h3, h4, h5, p, ul, ol, table, dl, blockquote";

function chunkTitlePath(sectionTitle, h3, h4) {
  return [sectionTitle, h3, h4].filter(Boolean).join(" › ");
}

function splitLong(text, size = 900, overlap = 120) {
  const clean = String(text).replace(/\s+/g, " ").trim();
  if (clean.length <= size) return [clean];
  const parts = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(clean.length, start + size);
    if (end < clean.length) {
      const dot = Math.max(clean.lastIndexOf(". ", end), clean.lastIndexOf("! ", end), clean.lastIndexOf("? ", end));
      if (dot > start + size * 0.5) end = dot + 1;
      else {
        const sp = clean.lastIndexOf(" ", end);
        if (sp > start + size * 0.5) end = sp;
      }
    }
    parts.push(clean.slice(start, end).trim());
    if (end >= clean.length) break;
    start = Math.max(start + 1, end - overlap);
  }
  return parts.filter(Boolean);
}

function collectChunks() {
  const chunks = [];
  const seen = new Set();
  // innerText даёт переносы строк по блокам, но есть не во всех движках — тогда textContent.
  const textOf = (el) => (el.innerText || el.textContent || "");

  const push = (title, raw, el, kind = "text") => {
    const text = String(raw || "").replace(/\u00a0/g, " ").replace(/[ \t]+/g, " ").trim();
    if (text.length < 40) return;
    const key = kind + "|" + text.slice(0, 180);
    if (seen.has(key)) return;
    seen.add(key);
    for (const piece of splitLong(text)) chunks.push({ title, text: piece, el, kind });
  };

  const sections = Array.from(document.querySelectorAll("main section"));

  // У части секций своего h2 нет (например, подразделы 4.2–4.6) — берём h2
  // ближайшей предыдущей секции, иначе в источниках светится сырой id вроде «t-word».
  const sectionTitleOf = (i) => {
    for (let j = i; j >= 0; j--) {
      const t = (sections[j].querySelector("h2")?.textContent || "").trim();
      if (t) return t;
    }
    return (sections[i].id || "Раздел").replace(/[-_]/g, " ");
  };

  sections.forEach((section, si) => {
    const sectionTitle = sectionTitleOf(si);
    let h3 = "";
    let h4 = "";
    let buffer = [];

    const flush = () => {
      if (!buffer.length) return;
      push(chunkTitlePath(sectionTitle, h3, h4), buffer.join("\n"), section, "text");
      buffer = [];
    };

    const walk = (parent) => {
      for (const el of Array.from(parent.children)) {
        if (el.matches(".prompt")) {
          flush();
          const path = chunkTitlePath(sectionTitle, h3, h4) || "Промпт";
          const pre = el.querySelector("pre");
          if (pre) push("Промпт: " + path, textOf(pre), el, "prompt");
          const items = el.querySelectorAll(".checkmenu li");
          if (items.length) {
            const list = Array.from(items).map((li) => "— " + textOf(li).trim()).join("\n");
            push("Чек-лист: " + chunkTitlePath(sectionTitle, h3, h4), list, el, "checklist");
          }
          continue;
        }
        if (el.matches("h2")) {
          flush();
          h3 = "";
          h4 = "";
          continue;
        }
        if (el.matches("h3")) {
          flush();
          h3 = el.textContent.trim();
          h4 = "";
          continue;
        }
        if (el.matches("h4, h5")) {
          flush();
          h4 = el.textContent.trim();
          continue;
        }
        if (el.matches(BLOCK_SELECTOR)) {
          buffer.push(textOf(el));
          continue;
        }
        walk(el);
      }
    };

    walk(section);
    flush();
  });

  return chunks;
}

/* ---------- запрос → контекст ---------- */

const MAX_CONTEXT_CHARS = 2400;

function retrieve(index, question) {
  if (!index || !index.chunks.length) return { context: "", sources: [], confident: false };

  const qTerms = Array.from(new Set(tokenize(question)));
  if (!qTerms.length) return { context: "", sources: [], confident: false };

  const scored = [];
  for (let i = 0; i < index.chunks.length; i++) {
    const s = index.scoreDoc(i, qTerms);
    if (s <= 0) continue;
    scored.push({ i, s: s + index.prefixBonus(i, qTerms) });
  }
  if (!scored.length) return { context: "", sources: [], confident: false };

  scored.sort((a, b) => b.s - a.s);
  const best = scored[0].s;
  // Порог уверенности: слишком слабые совпадения считаем «в методичке этого нет».
  const confident = best >= 2.2;
  const pool = scored.slice(0, 10).filter((x) => x.s >= Math.min(best * 0.35, 2.2));

  const picked = [];
  const usedTitles = new Set();
  let used = 0;
  for (const item of pool) {
    if (picked.length >= 4 || used >= MAX_CONTEXT_CHARS) break;
    const chunk = index.chunks[item.i];
    // Не берём два куска подряд из одного заголовка — лучше покрыть разные разделы.
    if (usedTitles.has(chunk.title) && picked.length >= 2) continue;
    usedTitles.add(chunk.title);
    picked.push({ ...chunk, score: item.s });
    used += chunk.text.length;
  }

  if (!picked.length) return { context: "", sources: [], confident: false };

  const context = picked
    .map((c, n) => `[${n + 1}] (раздел: ${c.title})\n${c.text}`)
    .join("\n\n")
    .slice(0, MAX_CONTEXT_CHARS);

  return { context, sources: picked.map((c) => ({ title: c.title, el: c.el })), confident };
}

/* ==================== ПРОМПТЫ ==================== */

const SYSTEM_PROMPT = `Ты — ИИ-навигатор по интерактивной методичке дисциплины ОП.03 «Информационные технологии» (специальность 09.02.07). Ты наставник, а не решебник.

Правила ответа:
1. Пиши по-русски, коротко и по делу, без воды.
2. Источник истины — пронумерованные фрагменты методички [1], [2], … Опирайся на них: держи номера практических занятий (ПЗ), названия заданий, формулировки критериев и пункты чек-листов.
3. Не выдумывай номера ПЗ, формулы, функции, пункты чек-листов и имена файлов. Нет данных во фрагментах — не додумывай.
4. Если фрагментов нет или они не по теме — честно напиши первой строкой: «В методичке этого нет.» и предложи, какой раздел открыть.
5. Не решай задание за студента. Проверяя работу, давай 3–5 пронумерованных пунктов вида: проблема → наводящий вопрос. Готовые формулы и пошаговые инструкции «жми сюда» не пиши.
6. На вопрос «что такое / где найти / как сдать» отвечай 2–4 предложениями, без списка.
7. Если сам вопрос студента — готовый промпт с ролью и форматом ответа, следуй его формату.
8. В конце одной строкой укажи источник: «Источник: [1] название раздела» (или «Источник: общие знания, в методичке не разбирается»).
9. Разметку markdown почти не используй: максимум **выделение** и обычные списки, без заголовков с #.`;

function buildUserTurn(question, retrieval) {
  if (!retrieval.context) {
    return `Релевантных фрагментов методички не найдено.\n\nВопрос студента:\n${question}\n\nСкажи, что в методичке этого нет, и предложи раздел, который стоит открыть.`;
  }
  return `Фрагменты методички:\n\n${retrieval.context}\n\nВопрос студента:\n${question}\n\nОтветь по правилам, опираясь только на эти фрагменты.`;
}

/* ==================== МИНИ-MARKDOWN ==================== */

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let list = null;
  let inCode = false;

  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      closeList();
      out.push(inCode ? "</pre>" : "<pre>");
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      out.push(line);
      continue;
    }
    const ul = line.match(/^\s*[-—•*]\s+(.*)$/);
    const ol = line.match(/^\s*(\d{1,2}[.)])\s+(.*)$/);
    if (ul) {
      if (list !== "ul") {
        closeList();
        out.push("<ul>");
        list = "ul";
      }
      out.push(`<li>${inlineMd(ul[1])}</li>`);
      continue;
    }
    if (ol) {
      if (list !== "ol") {
        closeList();
        out.push("<ol>");
        list = "ol";
      }
      out.push(`<li>${inlineMd(ol[2])}</li>`);
      continue;
    }
    closeList();
    if (!line.trim()) {
      out.push("<br>");
      continue;
    }
    out.push(`<p>${inlineMd(line)}</p>`);
  }
  closeList();
  if (inCode) out.push("</pre>");
  return out.join("");
}

/* ==================== СТИЛИ ==================== */

const css = `
  #ai-chat-widget {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, sans-serif;
  }
  .ai-chat-toggle {
    width: 58px;
    height: 58px;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    color: #08122a;
    background: linear-gradient(135deg, #4f8cff, #22c55e);
    box-shadow: 0 8px 24px rgba(79, 140, 255, .45);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform .15s, box-shadow .15s, opacity .15s;
  }
  .ai-chat-toggle:hover { transform: scale(1.06); }
  .ai-chat-toggle.hidden { opacity: 0; pointer-events: none; transform: scale(.9); }
  .ai-chat-window {
    position: absolute;
    right: 0;
    bottom: 0;
    width: min(420px, calc(100vw - 24px));
    height: min(600px, calc(100vh - 24px));
    display: none;
    flex-direction: column;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid #26385c;
    background: linear-gradient(180deg, #111c33, #16233f);
    box-shadow: 0 16px 48px rgba(0,0,0,.5);
  }
  .ai-chat-window.open { display: flex; }
  .ai-chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 12px 10px 16px;
    background: linear-gradient(135deg, #1c2f57, #13203c);
    border-bottom: 1px solid #26385c;
  }
  .ai-chat-header-text { flex: 1; min-width: 0; }
  .ai-chat-header h3 { margin: 0; font-size: 15px; color: #e8eefc; font-weight: 700; }
  .ai-chat-status {
    margin-top: 2px;
    font-size: 12px;
    color: #9fb0d0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ai-icon-btn {
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: #9fb0d0;
    font-size: 19px;
    line-height: 1;
    cursor: pointer;
    flex: none;
  }
  .ai-icon-btn:hover { background: #1a2949; color: #e8eefc; }
  .ai-chat-modelbar {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    background: #0e1a31;
    border-bottom: 1px solid #26385c;
  }
  .ai-chat-modelbar select {
    flex: 1;
    min-width: 0;
    background: #0b1220;
    color: #cfe0ff;
    border: 1px solid #26385c;
    border-radius: 9px;
    padding: 6px 8px;
    font: inherit;
    font-size: 12px;
    outline: none;
  }
  .ai-chat-modelbar button {
    border: 1px solid #26385c;
    background: #1e3358;
    color: #cfe0ff;
    border-radius: 9px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .ai-chat-modelbar button:hover { background: #264271; }
  .ai-chat-progress { height: 3px; background: #0b1428; }
  .ai-chat-progress span {
    display: block;
    height: 100%;
    width: 0;
    background: linear-gradient(90deg, #4f8cff, #22c55e);
    transition: width .2s;
  }
  .ai-chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .ai-chat-messages::-webkit-scrollbar { width: 6px; }
  .ai-chat-messages::-webkit-scrollbar-thumb { background: #2b4370; border-radius: 4px; }
  .ai-msg {
    max-width: 90%;
    padding: 10px 13px;
    border-radius: 14px;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .ai-msg.user {
    align-self: flex-end;
    background: linear-gradient(135deg, #4f8cff, #3a6ecc);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .ai-msg.assistant {
    align-self: flex-start;
    background: #0e1a31;
    border: 1px solid #26385c;
    color: #e8eefc;
    border-bottom-left-radius: 4px;
  }
  .ai-msg.assistant.md { white-space: normal; }
  .ai-msg.assistant.md p { margin: 4px 0; }
  .ai-msg.assistant.md ul, .ai-msg.assistant.md ol { margin: 6px 0 6px 20px; padding: 0; }
  .ai-msg.assistant.md li { margin: 3px 0; }
  .ai-msg.assistant.md code {
    background: #0b1428;
    border: 1px solid #26385c;
    border-radius: 5px;
    padding: 0 5px;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px;
  }
  .ai-msg.assistant.md pre {
    background: #0b1428;
    border: 1px solid #26385c;
    border-radius: 10px;
    padding: 10px;
    overflow-x: auto;
    white-space: pre-wrap;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px;
  }
  .ai-msg.error {
    align-self: stretch;
    max-width: 100%;
    background: rgba(239, 68, 68, .14);
    border: 1px solid #ef4444;
    color: #fca5a5;
  }
  .ai-msg.pending { color: #9fb0d0; font-style: italic; }
  .ai-sources {
    align-self: flex-start;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: -4px 0 0 2px;
  }
  .ai-sources button {
    background: #122546;
    border: 1px solid #2b4370;
    color: #9fc0ff;
    border-radius: 999px;
    padding: 3px 9px;
    font-size: 11.5px;
    cursor: pointer;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ai-sources button:hover { background: #1a3160; color: #dbe9ff; }
  .ai-copy-row { align-self: flex-start; margin: -6px 0 0 4px; }
  .ai-copy-row button {
    background: transparent;
    border: none;
    color: #6f86ad;
    font-size: 11.5px;
    cursor: pointer;
    padding: 2px 4px;
  }
  .ai-copy-row button:hover { color: #cfe0ff; text-decoration: underline; }
  .ai-chat-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0 12px 10px;
  }
  .ai-chat-chips button {
    background: #0e1a31;
    border: 1px solid #2b4370;
    color: #bcd0f5;
    border-radius: 999px;
    padding: 6px 11px;
    font-size: 12.5px;
    cursor: pointer;
  }
  .ai-chat-chips button:hover { background: #1a3160; border-color: #4f8cff; color: #fff; }
  .ai-chat-form {
    display: flex;
    gap: 8px;
    padding: 12px;
    border-top: 1px solid #26385c;
    background: #111c33;
  }
  .ai-chat-form textarea {
    flex: 1;
    min-height: 44px;
    max-height: 130px;
    resize: none;
    border-radius: 12px;
    border: 1px solid #26385c;
    background: #0b1220;
    color: #e8eefc;
    padding: 11px 12px;
    font: inherit;
    font-size: 14px;
    outline: none;
  }
  .ai-chat-form textarea:focus { border-color: #4f8cff; }
  .ai-chat-form button.ai-send {
    width: 44px;
    height: 44px;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    color: #08122a;
    background: linear-gradient(135deg, #4f8cff, #22c55e);
    display: flex;
    align-items: center;
    justify-content: center;
    flex: none;
  }
  .ai-chat-form button.ai-send.stop { background: linear-gradient(135deg, #f87171, #ef4444); color: #fff; }
  .ai-chat-form button.ai-send:disabled { opacity: .45; cursor: not-allowed; }
  .ai-ask-ai {
    background: #1e3358 !important;
    color: #cfe0ff !important;
    border: 1px solid #26385c !important;
  }
  .ai-ask-ai:hover { background: #264271 !important; }
`;

/* ==================== ХРАНЕНИЕ ==================== */

const LS_HISTORY = "ai-navigator-history-v1";
const LS_MODEL = "ai-navigator-model-v1";

function loadHistory() {
  try {
    const raw = JSON.parse(localStorage.getItem(LS_HISTORY) || "[]");
    return Array.isArray(raw)
      ? raw.filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string").slice(-20)
      : [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(LS_HISTORY, JSON.stringify(history.slice(-20)));
  } catch {
    /* приватный режим — просто не сохраняем */
  }
}

/* ==================== ВИДЖЕТ ==================== */

const QUICK_PROMPTS = [
  "Как оформить титульный лист?",
  "Где промпт для ПЗ по Excel?",
  "Что проверить перед сдачей работы?",
  "Чем отличается абсолютная ссылка от относительной?",
];

function mount() {
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.id = "ai-chat-widget";
  root.innerHTML = `
    <button class="ai-chat-toggle" type="button" aria-label="Открыть ИИ-чат">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
    </button>
    <div class="ai-chat-window" role="dialog" aria-label="ИИ-чат">
      <div class="ai-chat-header">
        <div class="ai-chat-header-text">
          <h3>ИИ-навигатор</h3>
          <div class="ai-chat-status">Нажмите, чтобы загрузить модель</div>
        </div>
        <button class="ai-icon-btn ai-clear" type="button" title="Очистить диалог" aria-label="Очистить диалог">⌫</button>
        <button class="ai-icon-btn ai-chat-close" type="button" aria-label="Закрыть">×</button>
      </div>
      <div class="ai-chat-modelbar">
        <select class="ai-model-select" aria-label="Выбор модели"></select>
        <button class="ai-reload" type="button" title="Перезагрузить выбранную модель">Сменить</button>
      </div>
      <div class="ai-chat-progress"><span></span></div>
      <div class="ai-chat-messages"></div>
      <div class="ai-chat-chips"></div>
      <form class="ai-chat-form">
        <textarea rows="1" placeholder="Спросите по методичке… (Enter — отправить)"></textarea>
        <button class="ai-send" type="submit" aria-label="Отправить">
          <svg class="ic-send" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
          <svg class="ic-stop" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="display:none">
            <rect x="6" y="6" width="12" height="12" rx="2"></rect>
          </svg>
        </button>
      </form>
    </div>
  `;
  document.body.appendChild(root);

  const toggle = root.querySelector(".ai-chat-toggle");
  const win = root.querySelector(".ai-chat-window");
  const closeBtn = root.querySelector(".ai-chat-close");
  const clearBtn = root.querySelector(".ai-clear");
  const status = root.querySelector(".ai-chat-status");
  const bar = root.querySelector(".ai-chat-progress span");
  const messagesEl = root.querySelector(".ai-chat-messages");
  const chipsEl = root.querySelector(".ai-chat-chips");
  const form = root.querySelector(".ai-chat-form");
  const input = form.querySelector("textarea");
  const sendBtn = form.querySelector(".ai-send");
  const icSend = sendBtn.querySelector(".ic-send");
  const icStop = sendBtn.querySelector(".ic-stop");
  const modelSelect = root.querySelector(".ai-model-select");
  const reloadBtn = root.querySelector(".ai-reload");

  const models = availableModels();
  let history = loadHistory();
  let engine = null;
  let loadedModel = "";
  let loading = false;
  let busy = false;
  let stopRequested = false;
  let index = null;

  if (!models.length) {
    setStatus("Модели недоступны");
    modelSelect.disabled = true;
    reloadBtn.disabled = true;
  } else {
    const saved = localStorage.getItem(LS_MODEL);
    for (const id of models) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = NICE_NAMES[id] || modelLabel(id);
      if (id === saved) opt.selected = true;
      modelSelect.appendChild(opt);
    }
  }

  /* ---------- служебное ---------- */

  function setStatus(text) {
    status.textContent = text;
    status.title = text;
  }

  function setBusyUI(on) {
    busy = on;
    sendBtn.classList.toggle("stop", on);
    sendBtn.setAttribute("aria-label", on ? "Остановить генерацию" : "Отправить");
    icSend.style.display = on ? "none" : "";
    icStop.style.display = on ? "" : "none";
  }

  function atBottom() {
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 90;
  }

  function scrollDown(force = false) {
    if (force || atBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMsg(text, role, extraClass = "") {
    const div = document.createElement("div");
    div.className = `ai-msg ${role}${extraClass ? " " + extraClass : ""}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollDown(true);
    return div;
  }

  function addSources(sources) {
    const useful = sources.filter((s) => s.el && document.contains(s.el));
    if (!useful.length) return;
    const row = document.createElement("div");
    row.className = "ai-sources";
    const seen = new Set();
    for (const s of useful) {
      if (seen.has(s.title)) continue;
      seen.add(s.title);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = s.title.length > 46 ? s.title.slice(0, 44) + "…" : s.title;
      btn.title = "Перейти к разделу: " + s.title;
      btn.addEventListener("click", () => {
        s.el.scrollIntoView({ behavior: "smooth", block: "start" });
        closeChat();
      });
      row.appendChild(btn);
    }
    messagesEl.appendChild(row);
    scrollDown();
  }

  function addCopyButton(getText) {
    const row = document.createElement("div");
    row.className = "ai-copy-row";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Скопировать ответ";
    btn.addEventListener("click", async () => {
      const text = getText();
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
        } catch {
          /* ignore */
        }
        document.body.removeChild(ta);
      }
      btn.textContent = "Скопировано ✓";
      setTimeout(() => (btn.textContent = "Скопировать ответ"), 1500);
    });
    row.appendChild(btn);
    messagesEl.appendChild(row);
  }

  function renderChips() {
    chipsEl.innerHTML = "";
    if (history.length) return;
    for (const q of QUICK_PROMPTS) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = q;
      b.addEventListener("click", () => sendMessage(q));
      chipsEl.appendChild(b);
    }
  }

  function rerenderMessages() {
    messagesEl.innerHTML = "";
    addMsg(
      "Привет! Я читаю эту методичку и отвечаю по её разделам. Могу подсказать, где лежит нужный промпт, разобрать работу по чек-листу и задать наводящие вопросы — но не решу задание за тебя.",
      "assistant"
    );
    for (const m of history) {
      if (m.role === "user") addMsg(m.content, "user");
      else {
        const div = addMsg("", "assistant", "md");
        div.innerHTML = renderMarkdown(m.content);
      }
    }
    renderChips();
  }

  /* ---------- загрузка модели ---------- */

  async function createEngine(modelId) {
    return CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        const pct = Math.max(0, Math.min(100, Math.round((report.progress || 0) * 100)));
        bar.style.width = pct + "%";
        setStatus(report.text ? `${modelLabel(modelId)} · ${pct}%` : `Загрузка ${pct}%`);
      },
    });
  }

  async function loadModel(preferredId) {
    if (!models.length) return null;
    const order = [];
    const wanted = preferredId || modelSelect.value || localStorage.getItem(LS_MODEL);
    if (wanted && models.includes(wanted)) order.push(wanted);
    for (const id of models) if (!order.includes(id)) order.push(id);

    let lastError = null;
    for (const id of order) {
      try {
        setStatus(`Загрузка ${modelLabel(id)}…`);
        const e = await createEngine(id);
        engine = e;
        loadedModel = id;
        modelSelect.value = id;
        try {
          localStorage.setItem(LS_MODEL, id);
        } catch {
          /* ignore */
        }
        bar.style.width = "100%";
        setStatus(`Готов · ${modelLabel(id)}`);
        return e;
      } catch (err) {
        lastError = err;
        console.warn("Модель не подошла:", id, err);
        engine = null;
      }
    }
    throw lastError || new Error("Нет доступных моделей");
  }

  async function ensureEngine(force = false) {
    if (engine && !force) return engine;
    if (loading) return engine;
    if (!navigator.gpu) {
      addMsg(
        "В этом браузере нет WebGPU. Откройте сайт в Chrome или Edge (через запустить.bat, а не двойным кликом по index.html).",
        "assistant",
        "error"
      );
      setStatus("WebGPU недоступен");
      return null;
    }

    loading = true;
    sendBtn.disabled = true;
    try {
      if (force && engine) {
        try {
          await engine.unload();
        } catch {
          /* ignore */
        }
        engine = null;
      }
      await loadModel(modelSelect.value);
      addMsg(`Модель ${modelLabel(loadedModel)} загружена. Спрашивайте по методичке.`, "assistant");
    } catch (err) {
      console.error(err);
      engine = null;
      addMsg(
        `Не удалось загрузить модель (${err.message || err}). Нужны Chrome/Edge и интернет на первый запуск — модель качается один раз и дальше работает офлайн.`,
        "assistant",
        "error"
      );
      setStatus("Ошибка загрузки");
    } finally {
      loading = false;
      if (!busy) sendBtn.disabled = false;
    }
    return engine;
  }

  /* ---------- генерация ---------- */

  async function sendMessage(text) {
    const question = String(text || "").trim();
    if (!question || busy) return;

    await ensureEngine();
    if (!engine) return;

    if (!index) {
      index = new SearchIndex(collectChunks());
      setStatus(`Готов · ${modelLabel(loadedModel)} · разделов: ${index.chunks.length}`);
    }

    setBusyUI(true);
    stopRequested = false;
    addMsg(question, "user");
    history.push({ role: "user", content: question });
    saveHistory(history);
    renderChips();

    const bubble = addMsg("", "assistant", "md pending");
    bubble.textContent = "Думаю…";
    setStatus("Думаю…");

    let reply = "";
    let retrieval = { context: "", sources: [], confident: false };

    try {
      retrieval = retrieve(index, question);
      const recent = history.slice(0, -1).slice(-4);
      const messages = [
        { role: "system", content: SYSTEM_PROMPT },
        ...recent,
        { role: "user", content: buildUserTurn(question, retrieval) },
      ];

      const stream = await engine.chat.completions.create({
        messages,
        stream: true,
        temperature: 0.4,
        top_p: 0.9,
        max_tokens: 700,
        enable_thinking: false,
      });

      bubble.textContent = "";
      bubble.classList.remove("pending");

      for await (const chunk of stream) {
        if (stopRequested) break;
        const delta = chunk.choices?.[0]?.delta?.content || "";
        if (!delta) continue;
        reply += delta;
        bubble.textContent = reply;
        scrollDown();
      }
    } catch (err) {
      console.error(err);
      if (!reply) {
        bubble.classList.add("error");
        bubble.classList.remove("pending", "md");
        bubble.textContent = `Ошибка ответа: ${err.message || err}. Попробуйте ещё раз или смените модель.`;
        setStatus("Ошибка");
        setBusyUI(false);
        sendBtn.disabled = false;
        input.focus();
        return;
      }
    }

    if (!reply.trim()) {
      bubble.classList.remove("md");
      bubble.classList.add("pending");
      bubble.textContent = stopRequested ? "Остановлено." : "Модель не ответила. Попробуйте переформулировать вопрос.";
    } else {
      bubble.innerHTML = renderMarkdown(reply);
      history.push({ role: "assistant", content: reply });
      saveHistory(history);
      addCopyButton(() => reply);
      if (!/источник/i.test(reply)) addSources(retrieval.sources);
    }

    setStatus(`Готов · ${modelLabel(loadedModel)}`);
    setBusyUI(false);
    sendBtn.disabled = false;
    input.focus();
  }

  function stopGeneration() {
    if (!busy) return;
    stopRequested = true;
    setStatus("Останавливаю…");
    engine?.interruptGenerate?.().catch?.(() => {});
  }

  /* ---------- события ---------- */

  toggle.addEventListener("click", () => openChat());
  closeBtn.addEventListener("click", closeChat);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && win.classList.contains("open")) closeChat();
  });

  clearBtn.addEventListener("click", () => {
    history = [];
    saveHistory(history);
    rerenderMessages();
    setStatus(`Готов · ${modelLabel(loadedModel || "")}`.trim());
  });

  reloadBtn.addEventListener("click", () => {
    if (loading) return;
    ensureEngine(true);
  });

  modelSelect.addEventListener("change", () => {
    if (engine && modelSelect.value !== loadedModel && !loading) ensureEngine(true);
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 130) + "px";
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (busy) return;
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) {
      stopGeneration();
      return;
    }
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(text);
  });

  /* ---------- кнопки «Спросить ИИ» у промптов ---------- */

  document.querySelectorAll("main .prompt").forEach((block) => {
    const actions = block.querySelector(".actions");
    const pre = block.querySelector("pre");
    if (!actions || !pre || block.querySelector(".ai-ask-ai")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ai-ask-ai";
    btn.textContent = "Спросить ИИ";
    btn.title = "Открыть чат и вставить этот промпт";
    btn.addEventListener("click", () => {
      openChat();
      input.value = (pre.innerText || pre.textContent || "").trim();
      input.dispatchEvent(new Event("input"));
      input.focus();
      const pos = input.value.indexOf("[вставь");
      if (pos > -1) input.setSelectionRange(pos, input.value.indexOf("]", pos) + 1);
      else input.setSelectionRange(input.value.length, input.value.length);
    });
    actions.appendChild(btn);
  });

  /* ---------- открытие ---------- */

  function openChat() {
    win.classList.add("open");
    toggle.classList.add("hidden");
    input.focus();
    if (!engine && !loading) ensureEngine();
  }

  function closeChat() {
    win.classList.remove("open");
    toggle.classList.remove("hidden");
  }

  // Публичный вход: window.aiChat.ask("вопрос") — удобно из кнопок страницы.
  window.aiChat = {
    open: openChat,
    close: closeChat,
    ask: (q) => sendMessage(q),
    fill: (q) => {
      openChat();
      input.value = String(q || "");
      input.dispatchEvent(new Event("input"));
      input.focus();
    },
  };

  rerenderMessages();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mount);
} else {
  mount();
}