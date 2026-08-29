const $ = (selector) => document.querySelector(selector);

const ui = {
  messages: $("#messages"), welcome: $("#welcome"), prompt: $("#prompt"), send: $("#send"),
  stop: $("#stop"), state: $("#agent-state"), name: $("#agent-name"), provider: $("#provider"),
  model: $("#model"), feed: $("#activity-feed"), stepCount: $("#step-count"),
  browserFrame: $("#browser-frame"), browserEmpty: $("#browser-empty"), browserUrl: $("#browser-url"),
  browserTitle: $("#browser-title"), browserDimensions: $("#browser-dimensions"),
  viewport: $("#browser-viewport"), cursor: $("#agent-cursor"), ripple: $("#click-ripple"),
  approval: $("#approval-modal"), approvalDescription: $("#approval-description"),
  approvalArgs: $("#approval-args"), sidebar: $(".sidebar"), workspace: $(".workspace-panel"),
  conversationList: $("#conversation-list"), modeButton: $("#mode-button"), modeLabel: $("#mode-label"),
  settings: $("#settings-modal"), profileSelect: $("#profile-select"),
  providerSelect: $("#provider-select"), modelInput: $("#model-input"),
  reasoningSelect: $("#reasoning-select"), providerNote: $("#provider-note"),
  apiKeyInput: $("#api-key-input"), apiKeyLabel: $("#api-key-label"),
};

let socket;
let running = false;
let assistantText = null;
let assistantMeta = null;
let activityCount = 0;
let approvalRequest = null;
let currentMode = "agent";
let activeConversation = null;
let appStatus = { profiles: [], openai_configured: false, anthropic_configured: false };

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.addEventListener("open", () => setStatus("Pronto para ajudar", true));
  socket.addEventListener("message", (event) => handleEvent(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    setStatus("Reconectando…", false);
    setTimeout(connect, 1500);
  });
}

async function loadStatus() {
  try {
    const status = await fetch("/api/status").then((response) => response.json());
    appStatus = status;
    ui.name.textContent = status.name;
    ui.provider.textContent = status.provider;
    ui.model.textContent = status.model;
    populateProfiles(status.profiles || []);
    ui.providerSelect.value = status.provider;
    ui.modelInput.value = status.model;
    updateProviderNote();
  } catch (_) { /* WebSocket status remains authoritative. */ }
}

function setStatus(text, online = true) {
  ui.state.textContent = text;
  $(".presence").style.background = online ? "var(--green)" : "var(--danger)";
}

function sendMessage(message = ui.prompt.value.trim()) {
  if (!message || running || !socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ action: "run", message, mode: currentMode, max_iterations: currentMode === "computer" ? 40 : 20 }));
  ui.prompt.value = "";
  resizeComposer();
}

function handleEvent(event) {
  const data = event.data || {};
  switch (event.type) {
    case "connected":
      setStatus("Pronto para ajudar", true);
      if (data.provider) ui.provider.textContent = data.provider;
      if (data.model) ui.model.textContent = data.model;
      if (data.reasoning_effort) ui.reasoningSelect.value = data.reasoning_effort;
      break;
    case "configured":
      ui.provider.textContent = data.provider; ui.model.textContent = data.model;
      ui.providerSelect.value = data.provider; ui.modelInput.value = data.model;
      ui.reasoningSelect.value = data.reasoning_effort || "medium";
      if (data.provider === "openai") appStatus.openai_configured = Boolean(data.credential_configured);
      if (data.provider === "anthropic") appStatus.anthropic_configured = Boolean(data.credential_configured);
      ui.apiKeyInput.value = "";
      closeSettings(); addActivity("◇", "Motor atualizado", `${data.provider} · ${data.model}`);
      break;
    case "conversation_list": renderConversations(data.conversations || [], data.active_id); break;
    case "history_loaded": loadHistory(data.messages || [], data.conversation_id); break;
    case "run_started":
      running = true; toggleRunning(true); hideWelcome(); appendUser(data.message); appendAssistant();
      addActivity("✦", "Tarefa iniciada", "Preparando contexto e ferramentas");
      break;
    case "iteration":
      setStatus(`Pensando · passo ${data.n}`, true);
      addActivity("◇", `Raciocinando · passo ${data.n}`, "Avaliando o próximo movimento");
      break;
    case "thought":
      if (data.text) addActivity("·", "Observação do agente", truncate(data.text, 110));
      break;
    case "tool_call":
      addActivity("→", toolLabel(data.name), describeArgs(data.args));
      if (data.name === "browser" || data.name === "computer_batch") openWorkspace();
      break;
    case "tool_result":
      addActivity(data.success ? "✓" : "×", data.success ? "Ferramenta concluída" : "Falha na ferramenta", truncate(data.output || "", 115), data.success ? "success" : "error");
      break;
    case "repair": addActivity("↻", "Chamada reparada", "O agente corrigiu o formato de uma ferramenta"); break;
    case "browser_ready":
      openWorkspace(); ui.browserDimensions.textContent = `${data.width} × ${data.height}`;
      addActivity("⌁", "Navegador iniciado", "Sessão isolada pronta para uso"); break;
    case "browser_frame": updateBrowser(data); break;
    case "browser_cursor": moveCursor(data); break;
    case "desktop_frame": updateDesktop(data); break;
    case "desktop_cursor": moveDesktopCursor(data); break;
    case "approval_request": showApproval(data); break;
    case "final": setAssistant(data.text || ""); break;
    case "cancelling": setStatus("Cancelando…", false); break;
    case "cancelled": setAssistant(data.message || "Tarefa cancelada."); break;
    case "error":
      addActivity("!", "Atenção", data.message || "Ocorreu um erro", "error");
      if (!assistantText && running) appendAssistant();
      if (running && assistantText && !assistantText.textContent.trim()) setAssistant(data.message || "Ocorreu um erro.");
      break;
    case "run_complete": finishRun(data); break;
    case "cleared": resetConversation(); break;
  }
}

function hideWelcome() { if (ui.welcome) ui.welcome.classList.add("hidden"); }

function appendUser(text) {
  const node = document.createElement("article"); node.className = "message user";
  node.innerHTML = `<div class="message-body">${formatText(text)}</div>`;
  ui.messages.appendChild(node); scrollMessages();
}

function appendAssistant(initialText = "") {
  const node = document.createElement("article"); node.className = "message assistant";
  node.innerHTML = `<div class="assistant-avatar">✦</div><div class="message-content"><div class="message-label">I.S.A.A.C.</div><div class="assistant-text">${initialText ? formatText(initialText) : '<span class="thinking"><i></i><i></i><i></i></span>'}</div><div class="message-meta"></div></div>`;
  ui.messages.appendChild(node); assistantText = node.querySelector(".assistant-text"); assistantMeta = node.querySelector(".message-meta"); scrollMessages();
}

function setAssistant(text) {
  if (!assistantText) appendAssistant();
  assistantText.innerHTML = formatText(text || "(sem resposta)"); scrollMessages();
}

function finishRun(data) {
  if (assistantText && !assistantText.textContent.trim()) setAssistant(data.output || "(sem resposta)");
  if (assistantMeta) assistantMeta.textContent = `${data.iterations || 0} passos · ${data.tool_calls || 0} ferramentas · ${reasonLabel(data.stopped_reason)}`;
  running = false; toggleRunning(false); setStatus(data.success ? "Pronto para ajudar" : "Tarefa encerrada", true);
  addActivity(data.success ? "✓" : "■", data.success ? "Tarefa concluída" : "Tarefa encerrada", reasonLabel(data.stopped_reason), data.success ? "success" : "error");
  assistantText = null; assistantMeta = null;
}

function toggleRunning(value) {
  ui.send.classList.toggle("hidden", value); ui.stop.classList.toggle("hidden", !value);
  ui.prompt.disabled = value;
}

function addActivity(icon, title, detail, state = "") {
  const placeholder = ui.feed.querySelector(".activity-placeholder"); if (placeholder) placeholder.remove();
  const item = document.createElement("div"); item.className = `activity-item ${state}`;
  item.innerHTML = `<div class="activity-icon">${escapeHtml(icon)}</div><div class="activity-copy"><strong>${escapeHtml(title)}</strong><span title="${escapeHtml(detail)}">${escapeHtml(detail)}</span></div>`;
  ui.feed.appendChild(item); ui.feed.scrollTop = ui.feed.scrollHeight;
  activityCount += 1; ui.stepCount.textContent = `${activityCount} ${activityCount === 1 ? "passo" : "passos"}`;
}

function updateBrowser(data) {
  if (data.image_base64) {
    ui.browserFrame.src = `data:${data.mime_type || "image/png"};base64,${data.image_base64}`;
    ui.browserFrame.style.display = "block"; ui.browserEmpty.classList.add("hidden"); ui.cursor.style.display = "flex";
  }
  ui.browserUrl.textContent = data.url || "about:blank"; ui.browserTitle.textContent = data.title || "Navegador";
  ui.browserDimensions.textContent = `${data.width || 1280} × ${data.height || 720}`;
  if (data.cursor) moveCursor({ ...data.cursor, width: data.width, height: data.height, action: "move" });
}

function updateDesktop(data) {
  openWorkspace();
  $("#workspace-title").textContent = "Computador";
  if (data.image_base64) {
    ui.browserFrame.src = `data:${data.mime_type || "image/png"};base64,${data.image_base64}`;
    ui.browserFrame.style.display = "block"; ui.browserEmpty.classList.add("hidden"); ui.cursor.style.display = "flex";
  }
  ui.browserUrl.textContent = "Computador local · tela real";
  ui.browserTitle.textContent = "Área de trabalho do Windows";
  ui.browserDimensions.textContent = `${data.width || 0} × ${data.height || 0}`;
  if (data.cursor) moveDesktopCursor({ ...data.cursor, ...data });
}

function moveDesktopCursor(data) {
  const left = Number(data.left || 0), top = Number(data.top || 0);
  moveCursor({
    x: Number(data.x) - left,
    y: Number(data.y) - top,
    width: Number(data.width || 1),
    height: Number(data.height || 1),
    action: data.action || "move",
  });
}

function moveCursor(data) {
  const x = Math.max(0, Math.min(100, (Number(data.x) / Number(data.width || 1280)) * 100));
  const y = Math.max(0, Math.min(100, (Number(data.y) / Number(data.height || 720)) * 100));
  ui.cursor.style.display = "flex"; ui.cursor.style.left = `${x}%`; ui.cursor.style.top = `${y}%`;
  if (data.action === "click" || data.action === "type") {
    ui.ripple.style.left = `${x}%`; ui.ripple.style.top = `${y}%`; ui.ripple.classList.remove("active");
    void ui.ripple.offsetWidth; ui.ripple.classList.add("active");
  }
}

function showApproval(data) {
  approvalRequest = data.request_id;
  const actions = data.args?.actions;
  ui.approvalDescription.textContent = Array.isArray(actions)
    ? `O agente quer executar ${actions.length} ação(ões) no computador. Revise o lote antes de permitir.`
    : `A ferramenta “${data.name}” tem nível de risco ${data.risk} e pediu autorização.`;
  ui.approvalArgs.textContent = JSON.stringify(data.args || {}, null, 2); ui.approval.classList.remove("hidden");
}

function resolveApproval(decision) {
  if (approvalRequest && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: "approval", request_id: approvalRequest, decision }));
  approvalRequest = null; ui.approval.classList.add("hidden");
}

function resetConversation() {
  [...ui.messages.querySelectorAll(".message")].forEach((node) => node.remove());
  ui.welcome?.classList.remove("hidden"); ui.feed.innerHTML = `<div class="activity-placeholder"><span>✦</span><p>As decisões e ferramentas do agente aparecerão aqui em tempo real.</p></div>`;
  activityCount = 0; ui.stepCount.textContent = "0 passos";
}

function loadHistory(messages, conversationId) {
  activeConversation = conversationId || null;
  [...ui.messages.querySelectorAll(".message")].forEach((node) => node.remove());
  if (!messages.length) { ui.welcome?.classList.remove("hidden"); return; }
  hideWelcome();
  for (const message of messages) {
    if (message.role === "user") appendUser(message.content || "");
    else appendAssistant(message.content || "");
  }
  assistantText = null; assistantMeta = null;
}

function renderConversations(conversations, activeId) {
  activeConversation = activeId || activeConversation;
  ui.conversationList.replaceChildren();
  for (const conversation of conversations) {
    const button = document.createElement("button"); button.type = "button";
    button.className = `conversation${conversation.id === activeConversation ? " active" : ""}`;
    const dot = document.createElement("span"); dot.className = "conversation-dot";
    const title = document.createElement("span"); title.textContent = conversation.title || "Nova conversa";
    button.append(dot, title);
    button.addEventListener("click", () => {
      if (!running && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: "select_chat", conversation_id: conversation.id }));
    });
    ui.conversationList.appendChild(button);
  }
}

function populateProfiles(profiles) {
  ui.profileSelect.replaceChildren();
  for (const profile of profiles) {
    const option = document.createElement("option"); option.value = `${profile.provider}|${profile.model}`;
    option.textContent = profile.label || `${profile.provider} · ${profile.model}`;
    ui.profileSelect.appendChild(option);
  }
  const custom = document.createElement("option"); custom.value = "custom"; custom.textContent = "Personalizado";
  ui.profileSelect.appendChild(custom);
}

function openSettings() {
  ui.providerSelect.value = ui.provider.textContent;
  ui.modelInput.value = ui.model.textContent;
  const match = (appStatus.profiles || []).find((p) => p.provider === ui.provider.textContent && p.model === ui.model.textContent);
  ui.profileSelect.value = match ? `${match.provider}|${match.model}` : "custom";
  updateProviderNote(); ui.settings.classList.remove("hidden");
}
function closeSettings() { ui.settings.classList.add("hidden"); }
function applyProfile() {
  if (ui.profileSelect.value === "custom") return;
  const [provider, ...modelParts] = ui.profileSelect.value.split("|");
  ui.providerSelect.value = provider; ui.modelInput.value = modelParts.join("|"); updateProviderNote();
}
function updateProviderNote() {
  const provider = ui.providerSelect.value;
  ui.apiKeyLabel.classList.toggle("hidden", !["openai", "anthropic"].includes(provider));
  if (provider === "openai") ui.providerNote.textContent = appStatus.openai_configured ? "OPENAI_API_KEY detectada no processo. O modo Computador usará o loop visual nativo do modelo." : "OPENAI_API_KEY não foi detectada. Configure-a no ambiente antes de usar este perfil.";
  else if (provider === "anthropic") ui.providerNote.textContent = appStatus.anthropic_configured ? "ANTHROPIC_API_KEY detectada no processo." : "ANTHROPIC_API_KEY não foi detectada. Configure-a no ambiente antes de usar este perfil.";
  else ui.providerNote.textContent = "Perfil local: nenhuma credencial é enviada para a interface.";
}
function saveSettings() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ action: "configure", provider: ui.providerSelect.value, model: ui.modelInput.value.trim(), reasoning_effort: ui.reasoningSelect.value, api_key: ui.apiKeyInput.value.trim() }));
}
function toggleMode() {
  if (running) return;
  currentMode = currentMode === "agent" ? "computer" : "agent";
  ui.modeLabel.textContent = currentMode === "computer" ? "Computador" : "Agente";
  ui.modeButton.classList.toggle("computer", currentMode === "computer");
  ui.prompt.placeholder = currentMode === "computer" ? "Diga o que fazer neste computador…" : "Converse com I.S.A.A.C…";
}

function openWorkspace() { if (window.innerWidth <= 900) ui.workspace.classList.add("open"); }
function scrollMessages() { ui.messages.scrollTop = ui.messages.scrollHeight; }
function resizeComposer() { ui.prompt.style.height = "auto"; ui.prompt.style.height = `${Math.min(ui.prompt.scrollHeight, 150)}px`; }
function truncate(text, n) { const value = String(text || "").replace(/\s+/g, " "); return value.length > n ? `${value.slice(0, n)}…` : value; }
function describeArgs(args) { const entries = Object.entries(args || {}); return entries.length ? truncate(entries.map(([k,v]) => `${k}: ${String(v)}`).join(" · "), 105) : "Sem parâmetros"; }
function toolLabel(name) { return ({ browser: "Usando o navegador", computer_batch: "Controlando o computador", computer_view: "Observando o computador", computer_describe: "Interpretando a tela", computer_control: "Controlando o computador", web_search: "Pesquisando na web", shell: "Executando comando", code: "Executando código", fs_read: "Lendo arquivo", fs_write: "Gravando arquivo" })[name] || `Usando ${name}`; }
function reasonLabel(reason) { return ({ final: "concluída", cancelled: "cancelada", approval_denied: "ação recusada", error: "erro", max_iterations: "limite de passos", budget_exhausted: "tempo esgotado", no_progress: "sem progresso" })[reason] || String(reason || "encerrada"); }
function escapeHtml(text) { return String(text ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[c]); }
function formatText(text) { return escapeHtml(text).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>"); }

ui.send.addEventListener("click", () => sendMessage());
ui.stop.addEventListener("click", () => { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: "cancel" })); });
ui.prompt.addEventListener("input", resizeComposer);
ui.prompt.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } });
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { ui.prompt.value = button.dataset.prompt; resizeComposer(); ui.prompt.focus(); }));
$("#new-chat").addEventListener("click", () => { if (!running && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: "new_chat" })); });
$("#mobile-menu").addEventListener("click", () => ui.sidebar.classList.toggle("open"));
$("#collapse-workspace").addEventListener("click", () => ui.workspace.classList.toggle("open"));
$("#approve").addEventListener("click", () => resolveApproval("approve_once"));
$("#deny").addEventListener("click", () => resolveApproval("deny"));
ui.modeButton.addEventListener("click", toggleMode);
$("#settings-button").addEventListener("click", openSettings); $("#model-button").addEventListener("click", openSettings);
$("#settings-close").addEventListener("click", closeSettings); $("#settings-cancel").addEventListener("click", closeSettings);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !ui.settings.classList.contains("hidden")) closeSettings();
});
$("#settings-save").addEventListener("click", saveSettings);
ui.profileSelect.addEventListener("change", applyProfile); ui.providerSelect.addEventListener("change", () => { ui.profileSelect.value = "custom"; updateProviderNote(); });
ui.modelInput.addEventListener("input", () => { ui.profileSelect.value = "custom"; });

loadStatus(); connect(); resizeComposer();
