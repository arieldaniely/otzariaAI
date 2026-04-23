(() => {
  "use strict";

  const INDEX_SCHEMA_VERSION = 1;
  const DB_NAME = "otzaria-ai-local-index";
  const DB_VERSION = 1;
  const CONTENT_PAGE_LIMIT = 5000;
  const DISCOVERY_LIMIT = 700;
  const SAVE_BATCH_SIZE = 60;
  const SEARCH_SCAN_LIMIT = 50000;
  const AUTOCOMPLETE_LIMIT = 8;
  const DEFAULT_PAGE_SIZE = 10;
  const DEFAULT_RESULT_POOL_SIZE = 500;

  const DEFAULT_SETTINGS = {
    topK: DEFAULT_RESULT_POOL_SIZE,
    pageSize: DEFAULT_PAGE_SIZE,
    idealWords: 54,
    maxWords: 72,
    overlapWords: 10,
    autoIndex: true,
    maxBooks: 0,
    maxCharsPerBook: 0,
  };

  const HEBREW_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת";
  const DISCOVERY_SEEDS = [
    "",
    ...HEBREW_LETTERS.split(""),
    "תורה",
    "משנה",
    "גמרא",
    "רמבם",
    "שולחן ערוך",
    "זוהר",
    "מדרש",
    "הלכה",
    "אגדה",
    "בראשית",
    "שבת",
  ];

  const HTML_TAG_RE = /<[^>]+>/g;
  const NIQQUD_RE = /[\u0591-\u05c7]/g;
  const NON_WORD_RE = /[^0-9A-Za-z\u0590-\u05ff"']+/g;
  const FINAL_LETTERS = new Map([
    ["ך", "כ"],
    ["ם", "מ"],
    ["ן", "נ"],
    ["ף", "פ"],
    ["ץ", "צ"],
  ]);
  const HEBREW_PREFIXES = [
    "וכש",
    "וש",
    "וה",
    "וב",
    "ול",
    "ומ",
    "כש",
    "שב",
    "שה",
    "מש",
    "מה",
    "ו",
    "ה",
    "ב",
    "ל",
    "מ",
    "ש",
    "כ",
  ];
  const STOP_WORD_LIST = [
    "של",
    "על",
    "אל",
    "כל",
    "לא",
    "כן",
    "אם",
    "או",
    "כי",
    "גם",
    "את",
    "עם",
    "זה",
    "זו",
    "הוא",
    "היא",
    "הם",
    "הן",
    "יש",
    "אין",
    "אשר",
    "רק",
    "עוד",
    "שם",
    "בו",
    "בה",
    "לו",
    "לה",
    "לי",
    "לך",
    "מן",
    "כמו",
    "וכו",
    "וכולי",
  ];
  const STOPWORDS = new Set(
    STOP_WORD_LIST.flatMap((word) => {
      const normalized = normalizeTermRaw(word);
      return [normalized, hebrewStem(normalized)];
    }),
  );

  const SYNONYM_SOURCE = {
    אהבה: ["חיבה", "רחמים", "דבקות", "חסד"],
    אמונה: ["בטחון", "ידיעה", "השגחה", "קבלה"],
    ברכה: ["שפע", "טובה", "הודאה"],
    גאולה: ["ישועה", "פדות", "קץ", "משיח"],
    דין: ["משפט", "חיוב", "הלכה", "עונש"],
    הלכה: ["דין", "פסק", "חיוב", "איסור", "היתר"],
    חכמה: ["בינה", "דעת", "תבונה", "שכל"],
    חסד: ["רחמים", "טובה", "צדקה", "אהבה"],
    יראה: ["פחד", "מורא", "אימה"],
    כפרה: ["מחילה", "סליחה", "תשובה"],
    מצוה: ["חיוב", "מעשה", "פקודה"],
    מלכות: ["ממשלה", "שלטון", "מלך"],
    נפש: ["נשמה", "רוח", "חיות"],
    צדקה: ["חסד", "נתינה", "עני"],
    קדושה: ["טהרה", "פרישות", "מקדש"],
    קרבן: ["זבח", "עולה", "מנחה"],
    שבת: ["מנוחה", "קידוש", "עונג"],
    תפילה: ["בקשה", "תחנונים", "עמידה", "ברכה"],
    תשובה: ["חרטה", "כפרה", "מחילה", "וידוי"],
    תורה: ["מקרא", "מצוה", "חכמה", "לימוד"],
  };

  const numberFormatter = new Intl.NumberFormat("he-IL");
  const state = {
    boot: null,
    db: null,
    settings: { ...DEFAULT_SETTINGS },
    indexState: initialIndexState(),
    indexing: false,
    cancelIndexing: false,
    currentQuery: "",
    currentResults: [],
    currentPage: 1,
    lastResults: new Map(),
    synonymMap: null,
  };
  const els = {};

  init();

  function init() {
    cacheElements();
    bindEvents();

    if (!window.Otzaria) {
      renderFatal("ה־SDK של אוצריא לא נטען. יש לפתוח את הקובץ כתוסף מתוך אוצריא.");
      return;
    }

    window.Otzaria.on("plugin.boot", handleBoot);
    window.Otzaria.on("theme.changed", applyTheme);
  }

  function cacheElements() {
    els.statusText = document.getElementById("statusText");
    els.indexPanel = document.querySelector(".index-panel");
    els.indexTitle = document.getElementById("indexTitle");
    els.indexMeta = document.getElementById("indexMeta");
    els.indexDetail = document.getElementById("indexDetail");
    els.indexProgress = document.getElementById("indexProgress");
    els.startIndex = document.getElementById("startIndex");
    els.pauseIndex = document.getElementById("pauseIndex");
    els.rebuildIndex = document.getElementById("rebuildIndex");
    els.searchForm = document.getElementById("searchForm");
    els.queryInput = document.getElementById("queryInput");
    els.searchButton = document.getElementById("searchButton");
    els.suggestions = document.getElementById("suggestions");
    els.resultSummary = document.getElementById("resultSummary");
    els.results = document.getElementById("results");
    els.pagination = document.getElementById("pagination");
  }

  function bindEvents() {
    els.searchForm.addEventListener("submit", handleSearch);
    els.startIndex.addEventListener("click", () => startIndexing({ rebuild: false }));
    els.pauseIndex.addEventListener("click", pauseIndexing);
    els.rebuildIndex.addEventListener("click", confirmRebuild);
    els.queryInput.addEventListener("input", debounce(handleAutocomplete, 160));
    els.results.addEventListener("click", handleResultAction);
    els.suggestions.addEventListener("click", handleSuggestionClick);
    els.pagination.addEventListener("click", handlePaginationClick);
  }

  async function handleBoot(payload) {
    state.boot = payload;
    applyTheme(payload.theme);
    setStatus("פותח אחסון מקומי...");

    try {
      state.db = await openIndexDb();
      state.settings = await loadSettings();
      state.indexState = await metaGet("indexState", initialIndexState());
      if (state.indexState.schemaVersion !== INDEX_SCHEMA_VERSION) {
        state.indexState = initialIndexState();
        await clearLocalIndex();
      }
      renderIndexState();

      if (state.settings.autoIndex && state.indexState.status !== "ready") {
        window.setTimeout(() => startIndexing({ rebuild: false }), 150);
      } else {
        setStatus(state.indexState.status === "ready" ? "האינדקס מוכן לחיפוש." : "האינדקס ממתין להפעלה.");
      }
    } catch (error) {
      renderFatal(`שגיאה באתחול התוסף: ${error.message}`);
    }
  }

  async function loadSettings() {
    if (!state.boot?.permissions?.includes("plugin.storage.read")) {
      return { ...DEFAULT_SETTINGS };
    }

    try {
      const stored = await sdk("storage.get", { key: "settings" });
      return { ...DEFAULT_SETTINGS, ...(stored || {}) };
    } catch {
      return { ...DEFAULT_SETTINGS };
    }
  }

  function applyTheme(theme) {
    if (!theme) return;
    const payload = theme.colorScheme ? theme : theme.theme;
    const cs = payload?.colorScheme;
    if (!cs) return;

    const root = document.documentElement;
    root.style.setProperty("--primary", cs.primary || "#1565c0");
    root.style.setProperty("--on-primary", cs.onPrimary || "#ffffff");
    root.style.setProperty("--surface", cs.surface || "#fbfcfe");
    root.style.setProperty("--surface-low", cs.surfaceContainerLow || cs.surface || "#ffffff");
    root.style.setProperty("--surface-high", cs.surfaceContainerHighest || cs.surfaceVariant || "#eef2f7");
    root.style.setProperty("--on-surface", cs.onSurface || "#172033");
    root.style.setProperty("--outline", cs.outline || "#c6ceda");
    root.style.setProperty("--error", cs.error || "#b3261e");

    if (payload.typography?.fontFamily) {
      root.style.setProperty("--font", `'${payload.typography.fontFamily}', system-ui, sans-serif`);
    }
  }

  async function startIndexing({ rebuild }) {
    if (state.indexing || !state.db) return;

    state.indexing = true;
    state.cancelIndexing = false;
    renderIndexState();

    try {
      if (rebuild) {
        await clearLocalIndex();
        state.indexState = initialIndexState();
      }

      await writeIndexState({
        status: "discovering",
        message: "מאתר ספרים בספריית אוצריא...",
        currentBookTitle: "",
      });

      let books = await getAllBooks();
      if (!books.length || rebuild) {
        books = await discoverBooks();
      }

      if (state.settings.maxBooks > 0) {
        books = books.slice(0, state.settings.maxBooks);
      }

      if (!books.length) {
        throw new Error("לא נמצאו ספרים דרך library.findBooks.");
      }

      await writeIndexState({
        status: "indexing",
        totalBooks: books.length,
        message: "בונה אינדקס מקומי...",
      });

      await indexBooks(books);

      if (!state.cancelIndexing) {
        await writeIndexState({
          status: "ready",
          progress: 100,
          currentBookTitle: "",
          message: "האינדקס מוכן לחיפוש.",
        });
        await notify("ui.showSuccess", "אינדקס אוצריא AI מוכן לחיפוש.");
      }
    } catch (error) {
      await writeIndexState({
        status: "error",
        message: error.message,
        currentBookTitle: "",
      });
      await notify("ui.showError", `שגיאה בבניית האינדקס: ${error.message}`);
    } finally {
      state.indexing = false;
      renderIndexState();
    }
  }

  function pauseIndexing() {
    state.cancelIndexing = true;
    setStatus("עוצר אחרי הספר הנוכחי...");
    els.pauseIndex.disabled = true;
  }

  async function confirmRebuild() {
    let confirmed = true;
    try {
      const res = await sdk("ui.showConfirm", {
        title: "לבנות אינדקס מחדש?",
        content: "האינדקס המקומי הקיים יימחק וייבנה מחדש מתוך ספריית אוצריא.",
      });
      confirmed = Boolean(res?.confirmed);
    } catch {
      confirmed = window.confirm("לבנות אינדקס מחדש?");
    }

    if (confirmed) {
      await startIndexing({ rebuild: true });
    }
  }

  async function discoverBooks() {
    const existing = await getAllBooks();
    const booksById = new Map(existing.map((book) => [book.bookId, book]));

    for (let i = 0; i < DISCOVERY_SEEDS.length; i += 1) {
      if (state.cancelIndexing) break;
      const seed = DISCOVERY_SEEDS[i];
      await writeIndexState({
        status: "discovering",
        progress: Math.round((i / DISCOVERY_SEEDS.length) * 12),
        totalBooks: booksById.size,
        message: "מאתר ספרים בספריית אוצריא...",
        currentBookTitle: seed ? `שאילתת גילוי: ${seed}` : "שאילתת גילוי כללית",
      });

      const found = await trySdk("library.findBooks", { query: seed, limit: DISCOVERY_LIMIT });
      addDiscoveredBooks(booksById, found);
      await sleep(35);
    }

    const recent = await trySdk("library.listRecentBooks", {});
    addDiscoveredBooks(booksById, recent);

    const books = [...booksById.values()].sort((a, b) => {
      const rankA = Number.isFinite(a.discoveryRank) ? a.discoveryRank : Number.MAX_SAFE_INTEGER;
      const rankB = Number.isFinite(b.discoveryRank) ? b.discoveryRank : Number.MAX_SAFE_INTEGER;
      return rankA - rankB;
    });
    await putBooks(books);
    await writeIndexState({
      totalBooks: books.length,
      progress: 14,
      message: `נמצאו ${formatNumber(books.length)} ספרים לאינדוקס.`,
      currentBookTitle: "",
    });
    return books;
  }

  function addDiscoveredBooks(booksById, rawBooks) {
    if (!Array.isArray(rawBooks)) return;
    rawBooks.forEach((raw, index) => {
      const book = normalizeBookMeta(raw);
      if (!book) return;
      const previous = booksById.get(book.bookId);
      const rawRank = Number(raw.order ?? raw.sortOrder ?? raw.index);
      const discoveryRank = previous?.discoveryRank ?? (Number.isFinite(rawRank) ? rawRank : booksById.size + index);
      booksById.set(book.bookId, {
        ...previous,
        ...book,
        discoveryRank,
        indexedAt: previous?.indexedAt || null,
        indexedSchemaVersion: previous?.indexedSchemaVersion || 0,
        chunkCount: previous?.chunkCount || 0,
      });
    });
  }

  async function indexBooks(books) {
    let indexedBooks = books.filter((book) => book.indexedSchemaVersion === INDEX_SCHEMA_VERSION).length;
    let chunkCount = state.indexState.chunkCount || 0;
    let tokenCount = state.indexState.tokenCount || 0;

    for (let i = 0; i < books.length; i += 1) {
      if (state.cancelIndexing) {
        await writeIndexState({
          status: "paused",
          indexedBooks,
          chunkCount,
          tokenCount,
          progress: computeProgress(indexedBooks, books.length),
          message: "האינדוקס נעצר. אפשר להמשיך מאותה נקודה.",
          currentBookTitle: "",
        });
        return;
      }

      const book = books[i];
      if (book.indexedSchemaVersion === INDEX_SCHEMA_VERSION) {
        continue;
      }

      await writeIndexState({
        status: "indexing",
        indexedBooks,
        totalBooks: books.length,
        chunkCount,
        tokenCount,
        progress: computeProgress(indexedBooks, books.length),
        message: "בונה אינדקס מקומי...",
        currentBookTitle: book.title,
      });

      const result = await indexBook(book);
      indexedBooks += 1;
      chunkCount += result.chunks;
      tokenCount += result.tokens;

      await putBook({
        ...book,
        indexedAt: Date.now(),
        indexedSchemaVersion: INDEX_SCHEMA_VERSION,
        chunkCount: result.chunks,
      });

      await writeIndexState({
        indexedBooks,
        totalBooks: books.length,
        chunkCount,
        tokenCount,
        avgChunkLength: chunkCount ? Math.round(tokenCount / chunkCount) : 60,
        progress: computeProgress(indexedBooks, books.length),
        currentBookTitle: book.title,
      });

      await sleep(20);
    }
  }

  async function indexBook(book) {
    const content = await fetchBookContent(book);
    const chunks = buildChunks(content, book);
    let savedChunks = 0;
    let savedTokens = 0;

    for (let i = 0; i < chunks.length; i += SAVE_BATCH_SIZE) {
      if (state.cancelIndexing) break;
      const batch = chunks.slice(i, i + SAVE_BATCH_SIZE);
      await saveChunkBatch(batch);
      savedChunks += batch.length;
      savedTokens += batch.reduce((sum, chunk) => sum + chunk.length, 0);
    }

    return { chunks: savedChunks, tokens: savedTokens };
  }

  async function fetchBookContent(book) {
    const parts = [];
    const signatures = new Set();
    let offset = 0;
    let totalChars = 0;

    for (let page = 0; page < 20000; page += 1) {
      if (state.cancelIndexing) break;

      const data = await sdk("library.getBookContent", {
        bookId: book.hostBookId ?? book.bookId,
        offset,
        limit: CONTENT_PAGE_LIMIT,
      });
      const text = normalizeContentResponse(data);
      if (!text) break;

      const signature = `${text.length}:${text.slice(0, 60)}:${text.slice(-60)}`;
      if (signatures.has(signature)) break;
      signatures.add(signature);

      parts.push(text);
      totalChars += text.length;

      if (state.settings.maxCharsPerBook > 0 && totalChars >= state.settings.maxCharsPerBook) {
        break;
      }
      if (text.length < CONTENT_PAGE_LIMIT - 30) {
        break;
      }

      offset += text.length;
      await sleep(8);
    }

    return parts.join("\n");
  }

  function normalizeContentResponse(data) {
    if (typeof data === "string") return data;
    if (Array.isArray(data)) {
      return data.map((item) => normalizeContentResponse(item)).filter(Boolean).join("\n");
    }
    if (data && typeof data === "object") {
      return normalizeContentResponse(data.content ?? data.text ?? data.body ?? data.lines ?? "");
    }
    return "";
  }

  function buildChunks(content, book) {
    const plain = stripHtml(content).replace(/\s+/g, " ").trim();
    if (!plain) return [];

    const words = plain.split(" ").filter(Boolean);
    const chunks = [];
    let cursor = 0;
    let chunkIndex = 0;

    while (cursor < words.length) {
      const maxEnd = Math.min(words.length, cursor + state.settings.maxWords);
      let end = maxEnd;

      for (let i = Math.min(words.length - 1, cursor + state.settings.idealWords); i < maxEnd; i += 1) {
        if (/[.:;?!]$/.test(words[i])) {
          end = i + 1;
          break;
        }
      }

      const rawText = words.slice(cursor, end).join(" ");
      const clean = cleanText(rawText);
      const terms = termCounts(clean);
      const termTotal = [...terms.values()].reduce((sum, count) => sum + count, 0);

      if (clean.length > 30 && termTotal >= 6) {
        chunks.push({
          chunkId: `${book.bookId}:${chunkIndex}`,
          bookId: book.bookId,
          hostBookId: book.hostBookId ?? book.bookId,
          title: book.title,
          ref: book.ref || "",
          bookRank: Number.isFinite(book.discoveryRank) ? book.discoveryRank : Number.MAX_SAFE_INTEGER,
          text: rawText,
          clean,
          length: termTotal,
          indexedAt: Date.now(),
          terms,
        });
        chunkIndex += 1;
      }

      if (end >= words.length) break;
      cursor = Math.max(cursor + 1, end - state.settings.overlapWords);
    }

    return chunks;
  }

  async function saveChunkBatch(chunks) {
    const postingMap = new Map();
    const storedChunks = chunks.map((chunk) => {
      chunk.terms.forEach((tf, term) => {
        if (!postingMap.has(term)) postingMap.set(term, []);
        postingMap.get(term).push({ c: chunk.chunkId, tf, len: chunk.length });
      });

      const { terms, ...stored } = chunk;
      return stored;
    });

    await appendChunksAndPostings(storedChunks, postingMap);
  }

  function appendChunksAndPostings(chunks, postingMap) {
    return new Promise((resolve, reject) => {
      const tx = state.db.transaction(["chunks", "postings", "terms"], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error || new Error("IndexedDB write failed"));
      tx.onabort = () => reject(tx.error || new Error("IndexedDB write aborted"));

      const chunkStore = tx.objectStore("chunks");
      const postingStore = tx.objectStore("postings");
      const termStore = tx.objectStore("terms");

      chunks.forEach((chunk) => chunkStore.put(chunk));

      postingMap.forEach((entries, term) => {
        const req = postingStore.get(term);
        req.onsuccess = () => {
          const record = req.result || { term, postings: [] };
          record.postings = record.postings.concat(entries);
          postingStore.put(record);
          termStore.put({ term, df: record.postings.length, updatedAt: Date.now() });
        };
      });
    });
  }

  async function handleSearch(event) {
    event.preventDefault();
    const query = els.queryInput.value.trim();
    if (!query) return;

    state.currentQuery = query;
    els.searchButton.disabled = true;
    hideSuggestions();
    renderLoading("מחפש באינדקס המקומי...");

    try {
      const results = await searchLocal(query, state.settings.topK);
      state.currentResults = results;
      state.currentPage = 1;
      renderCurrentPage();
    } catch (error) {
      renderError(`שגיאה בחיפוש: ${error.message}`);
    } finally {
      els.searchButton.disabled = false;
    }
  }

  async function searchLocal(query, limit) {
    const querySpecs = buildQueryTerms(query);
    if (!querySpecs.length) return [];

    const chunkCount = state.indexState.chunkCount || (await countStore("chunks"));
    if (!chunkCount) return [];

    const avgLen = Math.max(20, state.indexState.avgChunkLength || 60);
    const scores = new Map();
    const matchedTerms = new Map();

    for (const spec of querySpecs) {
      const record = await idbGet("postings", spec.term);
      const postings = record?.postings || [];
      if (!postings.length) continue;

      const df = postings.length;
      if (!spec.exact && df > chunkCount * 0.4) continue;

      const idf = Math.max(0.05, Math.log(1 + (chunkCount - df + 0.5) / (df + 0.5)));
      const maxScan = spec.exact ? SEARCH_SCAN_LIMIT : Math.floor(SEARCH_SCAN_LIMIT / 3);
      const stride = postings.length > maxScan ? Math.ceil(postings.length / maxScan) : 1;

      for (let i = 0; i < postings.length; i += stride) {
        const posting = postings[i];
        const tf = 1 + Math.log(posting.tf || 1);
        const bm25 = (tf * 2.2) / (tf + 1.2 * (0.35 + 0.65 * ((posting.len || avgLen) / avgLen)));
        const nextScore = (scores.get(posting.c) || 0) + spec.weight * idf * bm25;
        scores.set(posting.c, nextScore);

        if (!matchedTerms.has(posting.c)) matchedTerms.set(posting.c, new Set());
        matchedTerms.get(posting.c).add(spec.term);
      }
    }

    const candidateIds = [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, Math.max(limit * 8, 40))
      .map(([chunkId]) => chunkId);

    const chunks = await idbGetMany("chunks", candidateIds);
    const cleanQuery = cleanText(query);
    const exactTerms = new Set(tokenize(query));

    return chunks
      .map((chunk) => {
        const base = scores.get(chunk.chunkId) || 0;
        const matched = matchedTerms.get(chunk.chunkId) || new Set();
        const overlap = exactTerms.size ? [...exactTerms].filter((term) => matched.has(term)).length / exactTerms.size : 0;
        const phrase = cleanQuery && chunk.clean.includes(cleanQuery) ? 0.8 : 0;
        const proximity = estimateProximity(chunk.clean, exactTerms);
        const importance = Number.isFinite(chunk.bookRank) ? 1 / Math.sqrt(chunk.bookRank + 1) : 0;
        const score = base + overlap * 1.4 + phrase + proximity * 0.5 + importance * 0.18;

        return {
          ...chunk,
          score,
          features: { overlap, phrase, proximity, importance, matched: matched.size },
        };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  function buildQueryTerms(query) {
    const exact = tokenize(query);
    const specs = new Map();
    const synonyms = getSynonymMap();

    exact.forEach((term) => {
      specs.set(term, { term, weight: 1, exact: true });
      (synonyms.get(term) || []).forEach((synonym) => {
        if (!specs.has(synonym)) {
          specs.set(synonym, { term: synonym, weight: 0.46, exact: false });
        }
      });
    });

    return [...specs.values()];
  }

  function getSynonymMap() {
    if (state.synonymMap) return state.synonymMap;

    state.synonymMap = new Map();
    Object.entries(SYNONYM_SOURCE).forEach(([key, values]) => {
      const stemmedKey = hebrewStem(normalizeTermRaw(key));
      const stemmedValues = values
        .map((value) => hebrewStem(normalizeTermRaw(value)))
        .filter((value) => value && !STOPWORDS.has(value));
      state.synonymMap.set(stemmedKey, stemmedValues);

      stemmedValues.forEach((value) => {
        if (!state.synonymMap.has(value)) state.synonymMap.set(value, []);
        state.synonymMap.get(value).push(stemmedKey);
      });
    });
    return state.synonymMap;
  }

  function estimateProximity(clean, exactTerms) {
    if (exactTerms.size < 2) return 0;
    const words = tokenize(clean, { keepStopwords: true });
    const positions = [];

    words.forEach((word, index) => {
      if (exactTerms.has(word)) positions.push(index);
    });

    if (positions.length < 2) return 0;
    const span = positions[positions.length - 1] - positions[0] + 1;
    return Math.min(1, positions.length / span);
  }

  async function handleAutocomplete() {
    const value = els.queryInput.value.trim();
    if (value.length < 2 || !state.indexState.chunkCount) {
      hideSuggestions();
      return;
    }

    const words = cleanText(value).split(" ");
    const last = words[words.length - 1] || "";
    const prefix = hebrewStem(normalizeTermRaw(last));
    if (prefix.length < 2) {
      hideSuggestions();
      return;
    }

    const suggestions = await findTermsByPrefix(prefix, AUTOCOMPLETE_LIMIT);
    if (!suggestions.length) {
      hideSuggestions();
      return;
    }

    els.suggestions.innerHTML = suggestions
      .map((item) => {
        const label = displayTerm(item.term);
        return `<button type="button" data-suggest="${escapeAttr(label)}">${escapeHtml(label)}</button>`;
      })
      .join("");
    els.suggestions.hidden = false;
  }

  function handleSuggestionClick(event) {
    const button = event.target.closest("[data-suggest]");
    if (!button) return;

    const value = els.queryInput.value.trim();
    const parts = cleanText(value).split(" ");
    parts[parts.length - 1] = button.dataset.suggest;
    els.queryInput.value = parts.join(" ");
    hideSuggestions();
    els.queryInput.focus();
  }

  function hideSuggestions() {
    els.suggestions.hidden = true;
    els.suggestions.textContent = "";
  }

  function renderCurrentPage() {
    const total = state.currentResults.length;
    const pageSize = Math.max(5, Number(state.settings.pageSize) || DEFAULT_PAGE_SIZE);
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    state.currentPage = Math.max(1, Math.min(totalPages, state.currentPage || 1));

    const start = (state.currentPage - 1) * pageSize;
    const pageResults = state.currentResults.slice(start, start + pageSize);
    renderResults(pageResults, state.currentQuery, start);
    renderPagination(totalPages);
    renderResultSummary(total, start, pageResults.length);
  }

  function renderResults(results, query, offset = 0) {
    state.lastResults.clear();

    if (!results.length) {
      const suffix = state.indexState.status === "ready" ? "" : " האינדקס עדיין נבנה, ולכן ייתכן שתוצאות יופיעו בהמשך.";
      renderEmpty(`לא נמצאו תוצאות.${suffix}`);
      hidePagination();
      return;
    }

    els.results.innerHTML = results
      .map((result, index) => {
        const absoluteIndex = offset + index;
        const id = String(absoluteIndex);
        state.lastResults.set(id, result);
        return `
          <article class="result">
            <div class="result-head">
              <div>
                <div class="source">${escapeHtml(result.title || "ספר")}</div>
                <div class="location">${escapeHtml(result.ref || "")}</div>
              </div>
              <div class="score">#${absoluteIndex + 1} · ${escapeHtml(result.score.toFixed(2))}</div>
            </div>
            <p class="snippet">${highlightText(result.text, query)}</p>
            <div class="result-actions">
              <span class="features">${formatFeatures(result.features)}</span>
              <button class="secondary" type="button" data-open="${id}">פתח באוצריא</button>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function renderResultSummary(total, start, renderedCount) {
    if (!total) {
      els.resultSummary.hidden = true;
      els.resultSummary.textContent = "";
      return;
    }

    const from = start + 1;
    const to = start + renderedCount;
    els.resultSummary.hidden = false;
    els.resultSummary.textContent = `${formatNumber(total)} תוצאות מדורגות · מוצגות ${formatNumber(from)}-${formatNumber(to)}`;
  }

  function renderPagination(totalPages) {
    if (totalPages <= 1) {
      hidePagination();
      return;
    }

    const pages = paginationWindow(state.currentPage, totalPages);
    els.pagination.innerHTML = pages
      .map((page) => {
        if (page === "...") return `<span class="page-gap">...</span>`;
        const active = page === state.currentPage ? " active" : "";
        return `<button class="page-button${active}" type="button" data-page="${page}" aria-label="עמוד ${page}">${page}</button>`;
      })
      .join("");
    els.pagination.hidden = false;
  }

  function paginationWindow(page, totalPages) {
    const pages = new Set([1, 2, totalPages - 1, totalPages, page - 2, page - 1, page, page + 1, page + 2]);
    const normalized = [...pages].filter((item) => item >= 1 && item <= totalPages).sort((a, b) => a - b);
    const output = [];
    normalized.forEach((item, index) => {
      if (index > 0 && item - normalized[index - 1] > 1) output.push("...");
      output.push(item);
    });
    return output;
  }

  function hidePagination() {
    els.pagination.hidden = true;
    els.pagination.textContent = "";
    els.resultSummary.hidden = true;
  }

  function handlePaginationClick(event) {
    const button = event.target.closest("[data-page]");
    if (!button) return;
    state.currentPage = Number(button.dataset.page) || 1;
    renderCurrentPage();
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function handleResultAction(event) {
    const button = event.target.closest("[data-open]");
    if (!button) return;

    const result = state.lastResults.get(button.dataset.open);
    if (!result) return;

    button.disabled = true;
    try {
      await sdk("reader.openBook", {
        bookId: result.hostBookId ?? result.bookId,
        searchQuery: state.currentQuery,
      });
    } catch (error) {
      await notify("ui.showError", `לא הצלחתי לפתוח את הספר: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  }

  function formatFeatures(features) {
    const parts = [];
    if (features.overlap) parts.push(`חפיפה ${Math.round(features.overlap * 100)}%`);
    if (features.phrase) parts.push("ביטוי מלא");
    if (features.proximity) parts.push(`קרבה ${Math.round(features.proximity * 100)}%`);
    if (features.importance > 0.08) parts.push("ספר מרכזי");
    return parts.join(" · ");
  }

  function renderIndexState() {
    const info = state.indexState || initialIndexState();
    const indexed = info.indexedBooks || 0;
    const total = info.totalBooks || 0;
    const chunks = info.chunkCount || 0;
    const progress = info.progress ?? computeProgress(indexed, total);
    const ready = info.status === "ready";

    els.indexPanel.classList.toggle("ready", ready);
    els.indexTitle.textContent = ready ? "האינדקס מוכן" : indexTitleForStatus(info.status);
    els.indexMeta.textContent = `${formatNumber(indexed)} / ${formatNumber(total)} ספרים, ${formatNumber(chunks)} מקטעים`;
    els.indexDetail.textContent = info.currentBookTitle || info.message || "האינדקס נשמר בתוך התוסף.";
    els.indexProgress.style.width = `${Math.max(0, Math.min(100, progress))}%`;
    els.pauseIndex.hidden = !state.indexing;
    els.pauseIndex.disabled = state.cancelIndexing;
    els.startIndex.hidden = state.indexing;
    els.startIndex.textContent = ready ? "עדכן" : "המשך";
    els.rebuildIndex.disabled = state.indexing;
    setStatus(info.message || (ready ? "האינדקס מוכן לחיפוש." : "האינדקס עדיין לא הושלם."));
  }

  function indexTitleForStatus(status) {
    switch (status) {
      case "discovering":
        return "מאתר ספרים";
      case "indexing":
        return "בונה אינדקס";
      case "paused":
        return "האינדוקס נעצר";
      case "error":
        return "נדרשת בדיקה";
      default:
        return "מכין אינדקס מקומי";
    }
  }

  async function writeIndexState(patch) {
    state.indexState = {
      ...initialIndexState(),
      ...state.indexState,
      ...patch,
      schemaVersion: INDEX_SCHEMA_VERSION,
      updatedAt: Date.now(),
    };
    await metaSet("indexState", state.indexState);
    renderIndexState();
  }

  function initialIndexState() {
    return {
      schemaVersion: INDEX_SCHEMA_VERSION,
      status: "empty",
      message: "האינדקס עדיין לא נבנה.",
      progress: 0,
      totalBooks: 0,
      indexedBooks: 0,
      chunkCount: 0,
      tokenCount: 0,
      avgChunkLength: 60,
      currentBookTitle: "",
      updatedAt: 0,
    };
  }

  function computeProgress(indexedBooks, totalBooks) {
    if (!totalBooks) return 0;
    return Math.min(99, 14 + Math.round((indexedBooks / totalBooks) * 85));
  }

  function renderLoading(message) {
    hidePagination();
    els.results.innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  }

  function renderEmpty(message) {
    hidePagination();
    els.results.innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  }

  function renderError(message) {
    hidePagination();
    els.results.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
  }

  function renderFatal(message) {
    setStatus(message);
    renderError(message);
    els.searchButton.disabled = true;
    els.startIndex.disabled = true;
    els.rebuildIndex.disabled = true;
  }

  function setStatus(message) {
    els.statusText.textContent = message;
  }

  function openIndexDb() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains("meta")) {
          db.createObjectStore("meta", { keyPath: "key" });
        }
        if (!db.objectStoreNames.contains("books")) {
          db.createObjectStore("books", { keyPath: "bookId" });
        }
        if (!db.objectStoreNames.contains("chunks")) {
          const chunks = db.createObjectStore("chunks", { keyPath: "chunkId" });
          chunks.createIndex("bookId", "bookId", { unique: false });
        }
        if (!db.objectStoreNames.contains("postings")) {
          db.createObjectStore("postings", { keyPath: "term" });
        }
        if (!db.objectStoreNames.contains("terms")) {
          db.createObjectStore("terms", { keyPath: "term" });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("IndexedDB open failed"));
      request.onblocked = () => reject(new Error("IndexedDB blocked by another WebView"));
    });
  }

  async function clearLocalIndex() {
    await new Promise((resolve, reject) => {
      const tx = state.db.transaction(["meta", "books", "chunks", "postings", "terms"], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error || new Error("IndexedDB clear failed"));
      tx.onabort = () => reject(tx.error || new Error("IndexedDB clear aborted"));
      tx.objectStore("meta").clear();
      tx.objectStore("books").clear();
      tx.objectStore("chunks").clear();
      tx.objectStore("postings").clear();
      tx.objectStore("terms").clear();
    });
  }

  function idbRequest(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("IndexedDB request failed"));
    });
  }

  function idbTransactionDone(tx) {
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error || new Error("IndexedDB transaction failed"));
      tx.onabort = () => reject(tx.error || new Error("IndexedDB transaction aborted"));
    });
  }

  async function idbGet(storeName, key) {
    const tx = state.db.transaction(storeName, "readonly");
    return idbRequest(tx.objectStore(storeName).get(key));
  }

  function idbGetMany(storeName, keys) {
    return new Promise((resolve, reject) => {
      const tx = state.db.transaction(storeName, "readonly");
      const store = tx.objectStore(storeName);
      const results = new Array(keys.length);

      tx.oncomplete = () => resolve(results.filter(Boolean));
      tx.onerror = () => reject(tx.error || new Error("IndexedDB read failed"));
      keys.forEach((key, index) => {
        const req = store.get(key);
        req.onsuccess = () => {
          results[index] = req.result;
        };
      });
    });
  }

  async function metaGet(key, fallback = null) {
    const record = await idbGet("meta", key);
    return record ? record.value : fallback;
  }

  async function metaSet(key, value) {
    const tx = state.db.transaction("meta", "readwrite");
    tx.objectStore("meta").put({ key, value });
    await idbTransactionDone(tx);
  }

  async function getAllBooks() {
    const tx = state.db.transaction("books", "readonly");
    return (await idbRequest(tx.objectStore("books").getAll())) || [];
  }

  async function putBooks(books) {
    const tx = state.db.transaction("books", "readwrite");
    const store = tx.objectStore("books");
    books.forEach((book) => store.put(book));
    await idbTransactionDone(tx);
  }

  async function putBook(book) {
    const tx = state.db.transaction("books", "readwrite");
    tx.objectStore("books").put(book);
    await idbTransactionDone(tx);
  }

  async function countStore(storeName) {
    const tx = state.db.transaction(storeName, "readonly");
    return idbRequest(tx.objectStore(storeName).count());
  }

  function findTermsByPrefix(prefix, limit) {
    return new Promise((resolve, reject) => {
      const tx = state.db.transaction("terms", "readonly");
      const store = tx.objectStore("terms");
      const range = IDBKeyRange.bound(prefix, `${prefix}\uffff`);
      const request = store.openCursor(range);
      const terms = [];

      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor || terms.length >= limit * 4) {
          terms.sort((a, b) => (b.df || 0) - (a.df || 0));
          resolve(terms.slice(0, limit));
          return;
        }
        terms.push(cursor.value);
        cursor.continue();
      };
      request.onerror = () => reject(request.error || new Error("Autocomplete failed"));
    });
  }

  async function sdk(method, payload = {}) {
    const response = await window.Otzaria.call(method, payload);
    if (!response || response.success !== true) {
      const error = response?.error;
      throw new Error(error?.message || error?.code || `Otzaria.call failed: ${method}`);
    }
    return response.data;
  }

  async function trySdk(method, payload = {}) {
    try {
      return await sdk(method, payload);
    } catch (error) {
      console.warn(error);
      return null;
    }
  }

  async function notify(method, message) {
    try {
      await sdk(method, { message });
    } catch {
      setStatus(message);
    }
  }

  function normalizeBookMeta(raw) {
    if (!raw || typeof raw !== "object") return null;
    const rawId = raw.bookId ?? raw.id ?? raw.book_id ?? raw.key ?? raw.path ?? raw.title ?? raw.name;
    if (rawId === undefined || rawId === null || rawId === "") return null;

    const title = firstText(
      raw.title,
      raw.name,
      raw.displayName,
      raw.heTitle,
      raw.bookTitle,
      raw.ref,
      String(rawId),
    );

    return {
      bookId: String(rawId),
      hostBookId: rawId,
      title,
      ref: firstText(raw.ref, raw.reference, ""),
      author: firstText(raw.author, raw.authors, ""),
      discoveryRank: Number.isFinite(Number(raw.order ?? raw.sortOrder ?? raw.index))
        ? Number(raw.order ?? raw.sortOrder ?? raw.index)
        : null,
      indexedAt: null,
      indexedSchemaVersion: 0,
      chunkCount: 0,
    };
  }

  function firstText(...values) {
    for (const value of values) {
      if (Array.isArray(value) && value.length) return String(value.join(", "));
      if (typeof value === "string" && value.trim()) return value.trim();
      if (typeof value === "number") return String(value);
    }
    return "";
  }

  function termCounts(text) {
    const counts = new Map();
    tokenize(text).forEach((term) => {
      counts.set(term, (counts.get(term) || 0) + 1);
    });
    return counts;
  }

  function tokenize(text, options = {}) {
    return cleanText(text)
      .split(" ")
      .map(normalizeTermRaw)
      .map(hebrewStem)
      .filter((term) => term.length > 1 && (options.keepStopwords || !STOPWORDS.has(term)));
  }

  function cleanText(value) {
    if (!value) return "";
    return stripHtml(String(value))
      .replace(NIQQUD_RE, "")
      .replace(/[״]/g, '"')
      .replace(/[׳]/g, "'")
      .replace(NON_WORD_RE, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function stripHtml(value) {
    return String(value || "").replace(HTML_TAG_RE, " ");
  }

  function normalizeTermRaw(value) {
    return String(value || "")
      .toLowerCase()
      .replace(NIQQUD_RE, "")
      .replace(/[״׳'"]/g, "")
      .replace(/[ךםןףץ]/g, (letter) => FINAL_LETTERS.get(letter) || letter)
      .trim();
  }

  function displayTerm(term) {
    return String(term || "").replace(/[כמנפצ]$/, (letter) => {
      const finals = { כ: "ך", מ: "ם", נ: "ן", פ: "ף", צ: "ץ" };
      return finals[letter] || letter;
    });
  }

  function hebrewStem(word) {
    if (!word || word.length < 4) return word;
    for (const prefix of HEBREW_PREFIXES) {
      if (word.startsWith(prefix) && word.length > prefix.length + 2) {
        return word.slice(prefix.length);
      }
    }
    return word;
  }

  function highlightText(text, query) {
    const words = [...new Set(cleanText(query).split(" ").filter((word) => word.length > 1))].slice(0, 8);
    let escaped = escapeHtml(text);
    words.forEach((word) => {
      escaped = escaped.replace(new RegExp(`(${escapeRegExp(escapeHtml(word))})`, "gi"), "<mark>$1</mark>");
    });
    return escaped;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#96;");
  }

  function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function formatNumber(value) {
    return numberFormatter.format(Number(value) || 0);
  }

  function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  function debounce(fn, wait) {
    let timer = 0;
    return (...args) => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => fn(...args), wait);
    };
  }
})();
