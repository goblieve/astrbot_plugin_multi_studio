(() => {
  const page = window.AstrBotPluginPage;
  const list = document.querySelector("#profiles");
  const status = document.querySelector("#status");
  const addProfileButton = document.querySelector("#add-profile");
  const saveButton = document.querySelector("#save-profiles");
  const restoreButton = document.querySelector("#restore-profiles");
  const validateButton = document.querySelector("#validate-profiles");
  const diagnoseButton = document.querySelector("#refresh-diagnostics");
  const diagnostics = document.querySelector("#diagnostics-output");
  let loraAliases = [];
  let canRestore = false;
  const nav = document.querySelector("#profile-nav");
  const search = document.querySelector("#profile-search");
  const saveState = document.querySelector("#save-state");
  let busy = false;
  let loaded = false;
  let dirty = false;
  let activeCard = null;
  let workflows = [];
  const moveDialog = document.querySelector("#move-option-dialog");
  const moveTarget = document.querySelector("#move-option-target");
  const moveConfirm = document.querySelector("#move-option-confirm");
  const moveHint = document.querySelector("#move-option-hint");
  let movingRow = null;
  let destinations = [];


  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(text, className, onClick) {
    const node = element("button", className, text);
    node.type = "button";
    node.addEventListener("click", onClick);
    return node;
  }

  function markDirty() {
    if (status.textContent) showStatus("");
    if (!dirty) {
      dirty = true;
      updateButtons();
    }
  }

  function cards() {
    return [...list.querySelectorAll(".profile-card")];
  }

  function selectCard(card, focusNav = false) {
    activeCard = card;
    for (const item of cards()) item.hidden = item !== card;
    updateNavigation(focusNav);
  }

  function updateNavigation(focusActive = false) {
    const query = search.value.trim().toLocaleLowerCase();
    nav.replaceChildren();
    let activeButton = null;
    for (const card of cards()) {
      const name = card.querySelector(".profile-command").value.trim() || "未命名指令";
      const options = [...card.querySelectorAll(".option-name")];
      if (query && ![name, ...options.map((option) => option.value)].some(
        (item) => item.toLocaleLowerCase().includes(query),
      )) continue;
      const link = button("", "group-link", () => selectCard(card, true));
      const cmdText = card.querySelector(".profile-command").value.trim();
      const isCutout = cmdText === "抠图";
      const isUpscale = cmdText === "放大";
      const isMultiview = cmdText === "生成多视图";
      const mode = isCutout ? "直接抠图" : isUpscale ? "直接放大" : isMultiview ? "6 种多视图" : (card.querySelector(".profile-direct-editor").checked ? "直接编辑" : options.length + " 项");
      link.append(element("span", "group-name", name), element("span", "group-count", mode));
      if (card === activeCard) {
        link.setAttribute("aria-current", "true");
        activeButton = link;
      }
      nav.append(link);
    }
    if (!nav.children.length) {
      nav.append(element("p", "nav-empty", query ? "没有匹配的指令组" : "还没有指令组"));
    }
    if (focusActive) activeButton?.focus();
  }

  function optionExpanded(row, expanded) {
    row.querySelector(".option-body").hidden = !expanded;
    row.querySelector(".option-title").setAttribute("aria-expanded", String(expanded));
  }

  function updateButtons() {
    saveButton.disabled = !loaded || busy || !dirty;
    validateButton.disabled = !loaded || busy;
    restoreButton.disabled = !loaded || busy || !canRestore;
    addProfileButton.disabled = !loaded || busy;
    list.inert = busy;
    saveState.textContent = !loaded ? "配置尚未载入" : dirty ? "有未保存的更改" : "已与保存的配置同步";
    saveState.dataset.dirty = String(dirty);
  }

  function updateMetadata(result) {
    if (Array.isArray(result.lora_aliases)) loraAliases = result.lora_aliases;
    canRestore = result.can_restore === true;
    if (result.workflows) updateWorkflowList(result);
    if (result.multiview_prompts) renderMultiviewPrompts(result.multiview_prompts);
    updateButtons();
  }

  function duplicateName(value, names) {
    const existing = new Set(names.map((name) => name.trim().toLowerCase()));
    for (let index = 1; ; index += 1) {
      const suffix = index === 1 ? "副本" : "副本" + index;
      const candidate = [...value.trim()].slice(0, 40 - suffix.length).join("") + suffix;
      if (!existing.has(candidate.toLowerCase())) return candidate;
    }
  }

  function showStatus(message, kind = "") {
    status.textContent = message;
    status.dataset.kind = kind;
  }

  function createField(labelText, value, type, className, placeholder, maxLength) {
    const label = document.createElement("label");
    const title = document.createElement("span");
    title.textContent = labelText;
    const field = type === "textarea" ? document.createElement("textarea") : document.createElement("input");
    field.value = value ?? "";
    field.required = true;
    field.maxLength = maxLength;
    field.className = className;
    field.placeholder = placeholder;
    label.append(title, field);
    return { label, field };
  }

  function configuredOptionLoras(option) {
    if (Array.isArray(option.loras)) return option.loras;
    const alias = String(option.lora_alias || "").trim();
    return alias ? [{ alias, strength: option.lora_strength ?? 1.0 }] : [];
  }

  function addLoraRow(container, lora = {}) {
    const row = document.createElement("div");
    row.className = "lora-row";
    const aliasLabel = document.createElement("label");
    const aliasTitle = document.createElement("span");
    aliasTitle.textContent = "LoRA 别名";
    const select = document.createElement("select");
    select.className = "option-lora-alias";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = loraAliases.length ? "请选择 LoRA" : "请先配置 draw_lora_aliases";
    select.append(empty);
    const selectedAlias = String(lora.alias || lora.lora_alias || "").trim();
    const aliasExists = loraAliases.some((alias) => alias.toLocaleLowerCase() === selectedAlias.toLocaleLowerCase());
    if (selectedAlias && !aliasExists) {
      const missing = document.createElement("option");
      missing.value = selectedAlias;
      missing.textContent = `${selectedAlias}（未配置）`;
      select.append(missing);
    }
    for (const alias of loraAliases) {
      const item = document.createElement("option");
      item.value = alias;
      item.textContent = alias;
      select.append(item);
    }
    select.value = loraAliases.find((alias) => alias.toLocaleLowerCase() === selectedAlias.toLocaleLowerCase())
      ?? selectedAlias;
    select.required = true;
    aliasLabel.append(aliasTitle, select);

    const strength = createField(
      "强度（同时设置 M 和 C）",
      lora.strength ?? lora.lora_strength ?? 1.0,
      "input",
      "option-lora-strength",
      "每次调整 0.1",
      32,
    );
    strength.field.type = "number";
    strength.field.step = "0.1";
    strength.field.required = true;

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger small-button remove-lora";
    remove.textContent = "移除";
    remove.addEventListener("click", () => {
      row.remove();
      markDirty();
    });
    row.append(aliasLabel, strength.label, remove);
    container.append(row);
  }

  function addOptionRow(container, option = {}) {
    const row = document.createElement("div");
    row.className = "option-row";
    const head = document.createElement("div");
    head.className = "option-head";
    const title = button("", "option-title", () => {
      optionExpanded(row, title.getAttribute("aria-expanded") !== "true");
    });
    const actions = document.createElement("div");
    actions.className = "option-actions";
    const moveUp = document.createElement("button");
    moveUp.type = "button";
    moveUp.className = "small-button move-option move-option-up";
    moveUp.textContent = "↑";
    moveUp.title = "上移";
    moveUp.setAttribute("aria-label", "菜单选项上移");
    moveUp.addEventListener("click", () => moveOptionRow(row.parentElement, row, -1));
    const moveDown = document.createElement("button");
    moveDown.type = "button";
    moveDown.className = "small-button move-option move-option-down";
    moveDown.textContent = "↓";
    moveDown.title = "下移";
    moveDown.setAttribute("aria-label", "菜单选项下移");
    moveDown.addEventListener("click", () => moveOptionRow(row.parentElement, row, 1));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger small-button";
    remove.textContent = "删除选项";
    remove.addEventListener("click", () => {
      const currentList = row.parentElement;
      row.remove();
      updateOptionTitles(currentList);
      updateNavigation();
      markDirty();
    });
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "small-button";
    copy.textContent = "复制选项";
    copy.addEventListener("click", () => {
      const currentList = row.parentElement;
      const option = readOption(row);
      option.name = duplicateName(option.name, [...currentList.querySelectorAll(".option-name")].map((input) => input.value));
      addOptionRow(currentList, option);
      const duplicate = currentList.lastElementChild;
      row.after(duplicate);
      updateOptionTitles(currentList);
      optionExpanded(duplicate, true);
      updateNavigation();
      markDirty();
      duplicate.querySelector(".option-name").focus();
    });
    const relocate = button("移到…", "small-button relocate-option", () => openMoveDialog(row));
    actions.append(moveUp, moveDown, copy, relocate, remove);
    head.append(title, actions);
    const name = createField("显示给群友的选项名", option.name, "input", "option-name", "例如：去除路人", 40);
    const prompt = createField("此选项对应的图片编辑提示词", option.prompt, "textarea", "option-prompt", "写清楚修改目标，以及需要保留的内容", 4000);

    const loraConfig = document.createElement("div");
    loraConfig.className = "lora-config";
    const loraTitle = document.createElement("div");
    loraTitle.className = "lora-config-title";
    loraTitle.textContent = "关联的 LoRA（可添加多个）";
    const loraRows = document.createElement("div");
    loraRows.className = "lora-rows";
    for (const lora of configuredOptionLoras(option)) addLoraRow(loraRows, lora);
    const addLora = document.createElement("button");
    addLora.type = "button";
    addLora.className = "small-button add-lora";
    addLora.textContent = "＋ 添加 LoRA";
    addLora.disabled = loraAliases.length === 0;
    addLora.addEventListener("click", () => {
      addLoraRow(loraRows);
      markDirty();
      loraRows.lastElementChild.querySelector("select").focus();
    });
    const loraHelp = document.createElement("div");
    loraHelp.className = "lora-help";
    loraHelp.textContent = loraAliases.length
      ? "每个 LoRA 可单独设置强度；数字框上下箭头每次调整 0.1。"
      : "请先在插件配置 draw_lora_aliases 中添加别名。";

    const updateOptionControls = () => {
      const isUpscale = name.field.value.trim() === "放大";
      copy.disabled = isUpscale;
      prompt.field.required = !isUpscale;
      prompt.label.querySelector("span").textContent = isUpscale
        ? "提示词（放大选项不使用）"
        : "此选项对应的图片编辑提示词";
      prompt.field.placeholder = isUpscale
        ? "放大选项只控制工作流节点，无需填写提示词"
        : "写清楚修改目标，以及需要保留的内容";
      loraConfig.hidden = isUpscale;
      if (isUpscale) loraRows.replaceChildren();
      addLora.disabled = isUpscale || loraAliases.length === 0;
    };
    name.field.addEventListener("input", () => {
      updateOptionControls();
      updateOptionTitles(row.parentElement);
      updateNavigation();
    });
    updateOptionControls();
    loraConfig.append(loraTitle, loraRows, addLora, loraHelp);
    const fields = element("div", "option-body");
    fields.append(name.label, prompt.label, loraConfig);
    row.append(head, fields);
    optionExpanded(row, !option.name);
    container.append(row);
    updateOptionTitles(container);
  }

  function updateOptionTitles(container) {
    const rows = [...container.querySelectorAll(".option-row")];
    rows.forEach((row, index) => {
      const name = row.querySelector(".option-name").value.trim() || "未命名选项";
      row.querySelector(".option-title").textContent = (index + 1) + " · " + name;
      row.querySelector(".move-option-up").disabled = index === 0;
      row.querySelector(".move-option-down").disabled = index === rows.length - 1;
    });
  }

  function getMoveRestriction(card, row) {
    const options = [...card.querySelectorAll(".option-name")];
    if (options.length >= 30) return "已满";
    const name = row.querySelector(".option-name").value.trim().toLocaleLowerCase();
    if (name && options.some((option) => option.value.trim().toLocaleLowerCase() === name)) {
      return "有同名选项";
    }
    return "";
  }

  function openMoveDialog(row) {
    movingRow = row;
    const currentCard = row.closest(".profile-card");
    destinations = cards().filter((card) => card !== currentCard);
    const name = row.querySelector(".option-name").value.trim() || "未命名选项";
    document.querySelector("#move-option-name").textContent = name;
    moveTarget.replaceChildren(new Option("请选择目标指令组", ""));
    destinations.forEach((card, index) => {
      const command = card.querySelector(".profile-command").value.trim() || "未命名指令组";
      const count = card.querySelectorAll(".option-row").length;
      const restriction = getMoveRestriction(card, row);
      const label = (cards().indexOf(card) + 1) + " · " + command + "（" + (restriction || count + " 项") + "）";
      const option = new Option(label, String(index));
      option.disabled = Boolean(restriction);
      moveTarget.append(option);
    });
    const available = destinations.some((card) => !getMoveRestriction(card, row));
    moveTarget.disabled = !available;
    moveConfirm.disabled = true;
    if (!destinations.length) {
      moveHint.textContent = "还没有其他指令组。先添加一个指令组即可移动。";
    } else if (!available) {
      moveHint.textContent = "目标组已有同名选项或达到 30 项上限。";
    } else if (row.parentElement.children.length === 1) {
      moveHint.textContent = "移出后原指令组会变空。保存前请给原组添加选项，或删除该组。";
    } else {
      moveHint.textContent = "选项会移到目标组末尾，保存后生效。";
    }
    moveDialog.showModal();
  }

  moveTarget.addEventListener("change", () => { moveConfirm.disabled = moveTarget.value === ""; });
  document.querySelector("#move-option-cancel").addEventListener("click", () => moveDialog.close());
  moveDialog.addEventListener("close", () => {
    movingRow = null;
    destinations = [];
  });
  moveDialog.querySelector("form").addEventListener("submit", (event) => {
    event.preventDefault();
    const row = movingRow;
    const target = moveTarget.value === "" ? null : destinations[Number(moveTarget.value)];
    if (!row || !target || !row.isConnected || !target.isConnected || getMoveRestriction(target, row)) return;
    const original = row.parentElement;
    const destination = target.querySelector(".options-list");
    destination.append(row);
    updateOptionTitles(original);
    updateOptionTitles(destination);
    moveDialog.close();
    search.value = "";
    selectCard(target);
    markDirty();
    showStatus(original.children.length
      ? "选项已移动，保存后生效。"
      : "选项已移动。保存前请给原指令组添加选项，或删除空指令组。");
    row.querySelector(".option-title").focus();
  });

  function moveOptionRow(container, row, offset) {
    const rows = [...container.querySelectorAll(".option-row")];
    const index = rows.indexOf(row);
    const target = rows[index + offset];
    if (!target) return;
    if (offset < 0) {
      container.insertBefore(row, target);
    } else {
      container.insertBefore(target, row);
    }
    updateOptionTitles(container);
    markDirty();
  }

  function readMaskLoraDefaults(box) {
    return Object.fromEntries([...box.querySelectorAll(".mask-lora-section")].map((section) => {
      const overrides = [...section.querySelectorAll(".mask-lora-row")].flatMap((row) => {
        const baseline = row.loraDefault;
        const [enabled, model, clip] = row.querySelectorAll("input");
        const values = {
          enabled: enabled.checked,
          strength_model: Number.isFinite(model.valueAsNumber) ? model.valueAsNumber : null,
          strength_clip: Number.isFinite(clip.valueAsNumber) ? clip.valueAsNumber : null,
        };
        return Object.keys(values).some((key) => values[key] !== baseline[key]) ? [[baseline.name, values]] : [];
      });
      return [section.dataset.branch, Object.fromEntries(overrides)];
    }).filter(([, rows]) => Object.keys(rows).length));
  }

  function renderMaskLoraDefaults(box, catalog, defaults) {
    box.replaceChildren();
    if (!catalog || !Object.keys(catalog).length) return;
    box.append(element("h3", "", "默认 LoRA"),
      element("p", "hint", "重绘设置用于局部和整图重绘。移除设置用于只涂选区、提示词留空的任务。"));
    for (const [branch, label] of [["repaint", "重绘"], ["remove", "移除"]]) {
      if (!catalog[branch]?.length) continue;
      const section = element("details", "mask-lora-section");
      section.dataset.branch = branch;
      section.append(element("summary", "", label + " · " + catalog[branch].length + " 个 LoRA"));
      section.open = branch === "repaint";
      for (const item of catalog[branch]) {
        const override = defaults[branch]?.[item.name] || {};
        const row = element("div", "mask-lora-row");
        row.loraDefault = item;
        const modelName = element("span", "mask-lora-name", item.name.split("/").at(-1));
        modelName.title = item.name;
        const toggle = element("input");
        toggle.type = "checkbox";
        toggle.checked = override.enabled ?? item.enabled;
        const toggleLabel = element("label", "direct-editor-row");
        toggleLabel.append(toggle, element("span", "", "默认启用"));
        row.append(modelName, toggleLabel);
        for (const [key, title] of [["strength_model", "模型强度 M"], ["strength_clip", "CLIP 强度 C"]]) {
          const strength = element("input");
          strength.type = "number";
          strength.step = "0.1";
          strength.min = "-10";
          strength.max = "10";
          strength.value = override[key] ?? item[key];
          const field = element("label");
          field.append(element("span", "", title), strength);
          row.append(field);
        }
        section.append(row);
      }
      box.append(section);
    }
  }

  function addProfileCard(profile = {}) {
    const card = element("article", "profile-card");
    const header = element("header", "profile-summary");
    const title = element("h2", "profile-title");
    const actions = element("div", "profile-actions");
    actions.append(
      button("复制指令组", "", () => {
        const copy = readProfile(card);
        copy.command = duplicateName(copy.command, readProfiles().map((item) => item.command));
        search.value = "";
        const duplicate = addProfileCard(copy);
        card.after(duplicate);
        selectCard(duplicate);
        markDirty();
        duplicate.querySelector(".profile-command").focus();
      }),
      button("删除指令组", "danger", () => {
        if (!window.confirm("删除这个指令组？保存后才会生效。")) return;
        const next = card.nextElementSibling || card.previousElementSibling;
        card.remove();
        if (!cards().length) showEmpty();
        selectCard(next);
        markDirty();
      }),
    );
    header.append(title, actions);
    const command = createField("QQ 触发指令", profile.command, "input", "profile-command", "例如：改图、改二次图", 40);
    const updateTitle = () => {
      const value = command.field.value.trim();
      title.textContent = value || "未命名指令";
      updateNavigation();
    };
    command.field.addEventListener("input", updateTitle);

    const optionsBox = element("section", "options-box");
    const section = element("header", "section-head");
    const sectionActions = element("div", "actions");
    const options = element("div", "options-list");
    const expand = button("展开或收起", "", () => {
      const rows = [...options.querySelectorAll(".option-row")];
      const expandAll = rows.some((row) => row.querySelector(".option-body").hidden);
      for (const row of rows) optionExpanded(row, expandAll);
    });
    const addOption = button("＋ 添加选项", "", () => {
      addOptionRow(options);
      updateNavigation();
      markDirty();
      options.lastElementChild.querySelector(".option-name").focus();
    });
    sectionActions.append(expand, addOption);
    section.append(element("h3", "", "菜单选项"), sectionActions);
    for (const option of profile.options || []) addOptionRow(options, option);
    optionsBox.append(section, options);
    const workflowLabel = element("label");
    const workflowSelect = element("select", "profile-workflow");
    setWorkflowOptions(workflowSelect, profile.workflow_id || "");
    workflowLabel.append(element("span", "", "此指令组使用的工作流"), workflowSelect);
    workflowLabel.style.marginTop = "12px";
    const directLabel = element("label", "direct-editor-row");
    const directToggle = element("input", "profile-direct-editor");
    directToggle.type = "checkbox";
    directToggle.checked = profile.direct_editor === true;
    const fallbackLabel = element("label", "direct-editor-row");
    const fallbackToggle = element("input", "profile-gpt-image-fallback");
    fallbackToggle.type = "checkbox";
    fallbackToggle.checked = profile.allow_gpt_image_fallback === true;
    fallbackLabel.append(fallbackToggle, element("span", "", "ComfyUI 无法连通时，回退 GPT Image"));
    const fallbackHint = element("p", "hint", "需同时启用插件的 GPT Image 回退。回退时会发送所选提示词和参考图。");
    const loraBox = element("section", "options-box mask-lora-defaults");
    const loraDrafts = new Map([[workflowSelect.value, profile.mask_lora_defaults || {}]]);
    let loraWorkflow = workflowSelect.value;
    const updateLoraCatalog = () => {
      const selected = workflows.find((item) => item.id === workflowSelect.value);
      const catalog = selected?.mask_loras;
      renderMaskLoraDefaults(loraBox, catalog, loraDrafts.get(workflowSelect.value) || {});
      loraWorkflow = workflowSelect.value;
    };
    const updateMode = () => {
      const selected = workflows.find((item) => item.id === workflowSelect.value);
      const cmdText = command.field.value.trim().replace(/^\/+\s*/, "").toLowerCase();
      const isDirectBuiltin = [
        "抠图", "去背景", "抠背景", "rmbg",
        "放大", "高清修复", "超分", "upscale",
        "生成多视图", "多视图", "三视图", "multiview",
      ].includes(cmdText);
      fallbackToggle.disabled = isDirectBuiltin || selected?.kind === "mask";
      fallbackLabel.hidden = fallbackHint.hidden = fallbackToggle.disabled;
      workflowLabel.hidden = isDirectBuiltin;
      directLabel.hidden = isDirectBuiltin;
      directToggle.disabled = isDirectBuiltin || selected?.kind !== "mask";
      optionsBox.hidden = isDirectBuiltin || directToggle.checked;
      loraBox.hidden = isDirectBuiltin || !directToggle.checked || !loraBox.childElementCount;
    };
    command.field.addEventListener("input", updateMode);
    updateLoraCatalog();
    updateMode();
    directToggle.addEventListener("change", () => {
      updateMode();
      updateNavigation();
    });
    workflowSelect.addEventListener("change", () => {
      loraDrafts.set(loraWorkflow, readMaskLoraDefaults(loraBox));
      updateLoraCatalog();
      if (workflows.find((item) => item.id === workflowSelect.value)?.kind !== "mask") {
        directToggle.checked = false;
      }
      updateMode();
      updateNavigation();
    });
    workflowSelect.addEventListener("workflowcatalogchange", () => {
      loraDrafts.set(loraWorkflow, readMaskLoraDefaults(loraBox));
      updateLoraCatalog();
      updateMode();
    });
    directLabel.append(directToggle, element("span", "", "直接发送编辑链接，跳过菜单"));
    card.append(header, command.label, workflowLabel, directLabel, fallbackLabel, fallbackHint, loraBox, optionsBox);
    list.querySelector(".empty")?.remove();
    list.append(card);
    title.textContent = command.field.value.trim() || "未命名指令";
    return card;
  }

  function showEmpty() {
    list.append(element("p", "empty", "还没有改图指令。添加一个指令组即可开始。"));
  }

  function renderProfiles(profiles, selectedIndex = 0, expandedOptions = []) {
    list.replaceChildren();
    const rendered = profiles.map(addProfileCard);
    if (!rendered.length) showEmpty();
    const selected = rendered[selectedIndex] || rendered[0] || null;
    selectCard(selected);
    if (selected) [...selected.querySelectorAll(".option-row")].forEach((row, index) => {
      if (expandedOptions.includes(index)) optionExpanded(row, true);
    });
  }

  function readOption(row) {
    return {
      name: row.querySelector(".option-name").value.trim(),
      prompt: row.querySelector(".option-prompt").value.trim(),
      loras: [...row.querySelectorAll(".lora-row")].map((loraRow) => ({
        alias: loraRow.querySelector(".option-lora-alias").value.trim(),
        strength: loraRow.querySelector(".option-lora-strength").value.trim(),
      })),
    };
  }

  function readProfile(card) {
    const direct = card.querySelector(".profile-direct-editor").checked;
    const fallback = card.querySelector(".profile-gpt-image-fallback");
    return {
      command: card.querySelector(".profile-command").value.trim(),
      workflow_id: card.querySelector(".profile-workflow").value,
      direct_editor: direct,
      allow_gpt_image_fallback: !fallback.disabled && fallback.checked,
      ...(direct ? { mask_lora_defaults: readMaskLoraDefaults(card.querySelector(".mask-lora-defaults")) } : {}),
      options: [...card.querySelectorAll(".option-row")].map(readOption),
    };
  }

  function readProfiles() {
    return [...list.querySelectorAll(".profile-card")].map(readProfile);
  }

  async function load() {
    if (!page || typeof page.apiGet !== "function") {
      showStatus("请从 AstrBot 插件页面打开本界面。", "error");
      return;
    }
    try {
      await page.ready();
      const result = await page.apiGet("edit_options/state");
      loaded = true;
      updateMetadata(result);
      renderProfiles(Array.isArray(result.profiles) ? result.profiles : []);
      showStatus("");
    } catch (error) {
      showStatus(`读取配置失败：${error.message || error}`, "error");
    }
  }

  search.addEventListener("input", () => updateNavigation());
  list.addEventListener("input", markDirty);
  list.addEventListener("change", markDirty);
  addProfileButton.addEventListener("click", () => {
    search.value = "";
    const card = addProfileCard({ command: "", options: [{ name: "", prompt: "" }] });
    selectCard(card);
    markDirty();
    card.querySelector(".profile-command").focus();
  });

  for (const id of ["editor", "multiview", "workflows", "diagnostics"]) {
    document.querySelector("#show-" + id)?.addEventListener("click", () => {
      for (const view of ["editor", "multiview", "workflows", "diagnostics"]) {
        document.querySelector("#show-" + view)?.setAttribute("aria-pressed", String(view === id));
        const panel = document.querySelector("#" + view + "-panel");
        if (panel) panel.hidden = view !== id;
      }
      if (id === "workflows") refreshWorkflows();
    });
  }

  const multiviewFields = document.querySelector("#multiview-fields");
  const multiviewStatus = document.querySelector("#multiview-status");
  const multiviewSaveBtn = document.querySelector("#multiview-save");
  const multiviewResetBtn = document.querySelector("#multiview-reset");

  function renderMultiviewPrompts(prompts) {
    if (!multiviewFields || !prompts) return;
    multiviewFields.replaceChildren();

    const addSingleSection = (key, titleText, descText, openDefault = false) => {
      const sec = element("details", "mask-lora-section");
      sec.open = openDefault;
      sec.append(element("summary", "", titleText));
      if (descText) sec.append(element("p", "hint", descText));
      const ta = element("textarea", "mv-single");
      ta.dataset.key = key;
      ta.maxLength = 4000;
      ta.value = prompts[key] || "";
      sec.append(ta);
      multiviewFields.append(sec);
    };

    const addListSection = (key, titleText, descText) => {
      const sec = element("details", "mask-lora-section");
      const items = Array.isArray(prompts[key]) ? prompts[key] : [];
      sec.append(element("summary", "", `${titleText}（共 ${items.length} 条分镜提示词）`));
      if (descText) sec.append(element("p", "hint", descText));
      const grid = element("div", "options-list");
      items.forEach((val, idx) => {
        const lbl = element("label");
        lbl.append(element("span", "", `第 ${idx + 1} 张提示词`));
        const ta = element("textarea", "mv-list-item");
        ta.dataset.key = key;
        ta.dataset.index = String(idx);
        ta.maxLength = 4000;
        ta.value = val || "";
        lbl.append(ta);
        grid.append(lbl);
      });
      sec.append(grid);
      multiviewFields.append(sec);
    };

    addSingleSection(
      "three_view_prompt",
      "基础阶段：单图参考生成三视图提示词模板（方式 3、4、5 前置共用）",
      "第一阶段先把上传的单张参考图生成为正、侧、背三视图；随后方式 3、4、5 会同时参考 <image1> 原图与 <image2> 三视图。",
      true,
    );
    addSingleSection(
      "mode1_prompt",
      "方式 1：人物表情（偏动漫/偏浮夸 · 8格表情差分）",
      "单阶段直接由原图生成 16:9 4MP 八格动漫表情差分图。",
      true,
    );
    addSingleSection(
      "mode2_prompt",
      "方式 2：人物表情（偏写实 · 8格写实表情差分）",
      "单阶段直接由原图生成 16:9 4MP 八格写实表情差分图。",
      true,
    );
    addListSection(
      "mode3_prompts",
      "方式 3：动作差分（12 张 × 左右 2 姿势 = 24 姿势）",
      "生成三视图后依次执行以下 12 条动作差分提示词，最终拼合为「左侧三视图 + 右侧 4×3 动作总览大图」。",
    );
    addListSection(
      "mode4_prompts",
      "方式 4：多角度生成（12 个机位角度）",
      "生成三视图后依次执行以下 12 条机位提示词，最终拼合为「左侧三视图 + 右侧 4×3 多角度总览大图」。",
    );
    addSingleSection(
      "mode5_prompt",
      "方式 5：动作/多视图单独生成（自定义单图默认提示词）",
      "用户在 QQ 选择 5 且未在消息里附带“5=自定义要求”时，使用此默认提示词生成 1 张单图并与三视图并列输出。",
      true,
    );
    addListSection(
      "mode6_prompts",
      "方式 6：多角度里图生成（12 个机位角度）",
      "生成三视图后依次执行以下 12 条机位提示词，最终拼合为「左侧三视图 + 右侧 4×3 多角度总览大图」。",
    );
  }

  function readMultiviewPrompts() {
    const out = {};
    for (const ta of multiviewFields.querySelectorAll("textarea.mv-single")) {
      out[ta.dataset.key] = ta.value.trim();
    }
    for (const key of ["mode3_prompts", "mode4_prompts", "mode6_prompts"]) {
      const nodes = [...multiviewFields.querySelectorAll(`textarea.mv-list-item[data-key="${key}"]`)];
      nodes.sort((a, b) => Number(a.dataset.index) - Number(b.dataset.index));
      out[key] = nodes.map((n) => n.value.trim());
    }
    return out;
  }

  multiviewSaveBtn?.addEventListener("click", async () => {
    if (!loaded) return;
    multiviewSaveBtn.disabled = true;
    multiviewStatus.textContent = "正在保存多视图提示词…";
    try {
      const res = await page.apiPost("edit_options/multiview/save", { prompts: readMultiviewPrompts() });
      if (res.multiview_prompts) renderMultiviewPrompts(res.multiview_prompts);
      multiviewStatus.textContent = "✅ 多视图提示词已保存并立即生效。";
    } catch (error) {
      multiviewStatus.textContent = "❌ 保存失败：" + (error.message || String(error));
    } finally {
      multiviewSaveBtn.disabled = false;
    }
  });

  let multiviewResetArmed = false;
  let multiviewResetTimer = null;
  const disarmMultiviewReset = () => {
    multiviewResetArmed = false;
    if (multiviewResetTimer) {
      clearTimeout(multiviewResetTimer);
      multiviewResetTimer = null;
    }
    if (multiviewResetBtn) multiviewResetBtn.textContent = "恢复默认提示词";
  };

  multiviewResetBtn?.addEventListener("click", async () => {
    if (!loaded) return;
    if (!multiviewResetArmed) {
      multiviewResetArmed = true;
      multiviewResetBtn.textContent = "再次点击确认恢复";
      multiviewStatus.textContent = "⚠️ 再次点击「再次点击确认恢复」将把全部 6 种多视图提示词重置为内置默认值（6 秒后自动取消）。";
      multiviewResetTimer = setTimeout(disarmMultiviewReset, 6000);
      return;
    }
    disarmMultiviewReset();
    multiviewResetBtn.disabled = true;
    multiviewStatus.textContent = "正在恢复默认提示词…";
    try {
      const res = await page.apiPost("edit_options/multiview/reset", {});
      if (res.multiview_prompts) renderMultiviewPrompts(res.multiview_prompts);
      multiviewStatus.textContent = "✅ 已恢复为默认多视图提示词。";
    } catch (error) {
      multiviewStatus.textContent = "❌ 恢复失败：" + (error.message || String(error));
    } finally {
      multiviewResetBtn.disabled = false;
    }
  });

  const workflowList = document.querySelector("#workflow-library-list");
  const workflowStatus = document.querySelector("#workflow-status");
  const workflowFile = document.querySelector("#workflow-file");
  const workflowKind = document.querySelector("#workflow-kind");
  const workflowName = document.querySelector("#workflow-name");
  const workflowExisting = document.querySelector("#workflow-existing");
  const workflowBindings = document.querySelector("#workflow-bindings");
  const workflowSave = document.querySelector("#workflow-save");
  let workflowGraph = null;
  let workflowSaving = false;
  const rolesByKind = {
    qwen: {positive:"正向提示词", negative:"负向提示词", width:"宽度", height:"高度",
      latent:"采样器 latent_image 输入", output_image:"成品节点的 images 输入"},
    mask: {image:"原图加载节点的文件输入", mask:"黑白遮罩加载节点的文件输入",
      positive:"编辑提示词", output:"成品输出节点"},
  };

  function setWorkflowOptions(select, current) {
    select.replaceChildren(new Option("默认 Qwen 工作流", ""));
    for (const item of workflows) {
      select.append(new Option(item.name + (item.kind === "mask" ? "（局部重绘）" : ""), item.id));
    }
    if (current && !workflows.some((item) => item.id === current)) {
      select.append(new Option("绑定的工作流不可用", current));
    }
    select.value = current;
  }

  function updateWorkflowList(result) {
    workflows = Array.isArray(result.workflows) ? result.workflows : [];
    for (const select of list.querySelectorAll(".profile-workflow")) {
      setWorkflowOptions(select, select.value);
      select.dispatchEvent(new Event("workflowcatalogchange"));
    }
    const editing = workflowExisting.value;
    workflowExisting.replaceChildren(new Option("新增工作流", ""));
    for (const item of workflows) workflowExisting.append(new Option(item.name, item.id));
    workflowExisting.value = workflows.some((item) => item.id === editing) ? editing : "";
    workflowList.replaceChildren();
    for (const item of workflows) {
      const card = element("div", "sidebar-top health-item");
      const description = element("div");
      description.append(element("h3", "", item.name),
        element("p", "", (item.kind === "mask" ? "局部重绘" : "Qwen 普通改图") + " · " + item.nodes + " 个节点"));
      const remove = button("删除", "danger", async () => {
        if (!window.confirm("删除「" + item.name + "」？如果还有指令组绑定，需要先解除绑定。")) return;
        remove.disabled = true;
        try {
          updateWorkflowList(await page.apiPost("edit_options/workflows/delete", {id:item.id}));
          workflowStatus.textContent = "已删除。";
        } catch (error) {
          workflowStatus.textContent = error.message || String(error);
          remove.disabled = false;
        }
      });
      card.append(description, remove);
      workflowList.append(card);
    }
    if (!workflows.length) workflowList.append(element("p", "hint", "工作流库为空。"));
    if (typeof result.mask_editor_ready === "boolean") {
      document.querySelector("#mask-library-state").textContent = result.mask_editor_ready
        ? "局部重绘网页服务已启动。" : "局部重绘网页服务待配置 HTTPS 地址。";
    }
  }

  async function refreshWorkflows() {
    try {
      if (!loaded) return;
      updateWorkflowList(await page.apiGet("edit_options/workflows"));
    } catch (error) {
      workflowStatus.textContent = error.message || String(error);
    }
  }
  document.querySelector("#refresh-workflows").addEventListener("click", refreshWorkflows);

  function buildBindingFields() {
    workflowBindings.replaceChildren();
    workflowSave.disabled = !workflowGraph || workflowSaving;
    if (!workflowGraph) return;
    for (const [role, hint] of Object.entries(rolesByKind[workflowKind.value])) {
      const label = element("label");
      const select = element("select");
      select.required = true;
      select.dataset.role = role;
      select.append(new Option("请选择节点" + (role === "output" ? "" : "与字段"), ""));
      const preferred = [];
      for (const [node, value] of Object.entries(workflowGraph)) {
        const fields = role === "output" ? [null] : Object.keys(value.inputs);
        for (const input of fields) {
          const hasBinding = value._meta?.hub?.[role] === input;
          if (role === "output" && !hasBinding && value.class_type !== "SaveImage" && value.class_type !== "PreviewImage") continue;
          const token = JSON.stringify({node, input});
          select.append(new Option(node + " · " + value.class_type + (input ? " · " + input : ""), token));
          if (hasBinding) preferred.push(token);
        }
      }
      if (preferred.length === 1) select.value = preferred[0];
      label.append(element("span", "", hint), select);
      workflowBindings.append(label);
    }
  }

  workflowFile.addEventListener("change", async () => {
    const file = workflowFile.files[0];
    workflowGraph = null;
    buildBindingFields();
    if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw new Error("工作流文件最大 1 MiB。");
      const graph = JSON.parse(await file.text());
      if (workflowFile.files[0] !== file) return;
      const values = Object.values(graph || {});
      if (!values.length || values.length > 400 || values.some((node) =>
        !node || typeof node.class_type !== "string" || !node.inputs || typeof node.inputs !== "object"
      )) throw new Error("请选择节点完整的 ComfyUI API 格式 JSON。");
      workflowGraph = graph;
      if (values.some((node) => typeof node._meta?.hub?.mask === "string")) workflowKind.value = "mask";
      if (!workflowName.value) workflowName.value = file.name.replace(/\.json$/i, "").slice(0, 40);
      buildBindingFields();
      workflowStatus.textContent = "请确认每项绑定后保存。";
    } catch (error) {
      workflowStatus.textContent = error.message || String(error);
    }
  });
  workflowKind.addEventListener("change", buildBindingFields);
  workflowExisting.addEventListener("change", () => {
    const item = workflows.find((workflow) => workflow.id === workflowExisting.value);
    if (!item) return;
    workflowName.value = item.name;
    workflowKind.value = item.kind;
    buildBindingFields();
  });
  document.querySelector("#workflow-import").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (workflowSaving || !workflowGraph) return;
    const bindings = Object.fromEntries([...workflowBindings.querySelectorAll("select")].map(
      (select) => [select.dataset.role, JSON.parse(select.value)],
    ));
    workflowSaving = true;
    workflowSave.disabled = true;
    try {
      const result = await page.apiPost("edit_options/workflows/save", {
        id:workflowExisting.value, name:workflowName.value, kind:workflowKind.value, graph:workflowGraph, bindings,
      });
      updateWorkflowList(result);
      workflowFile.value = "";
      workflowName.value = "";
      workflowExisting.value = "";
      workflowGraph = null;
      buildBindingFields();
      workflowStatus.textContent = "已保存。可返回指令组选择这个工作流。";
    } catch (error) {
      workflowStatus.textContent = error.message || String(error);
    } finally {
      workflowSaving = false;
      workflowSave.disabled = !workflowGraph;
    }
  });

  async function submitProfiles(action, payload) {
    if (busy) return;
    const selectedIndex = cards().indexOf(activeCard);
    const expandedOptions = activeCard ? [...activeCard.querySelectorAll(".option-row")]
      .flatMap((row, index) => row.querySelector(".option-body").hidden ? [] : [index]) : [];
    busy = true;
    updateButtons();
    showStatus("正在处理…");
    try {
      const result = await page.apiPost("edit_options/" + action, payload);
      if (action === "validate") {
        showStatus("校验通过，可以保存。", "success");
      } else {
        dirty = false;
        updateMetadata(result);
        renderProfiles(result.profiles, selectedIndex, action === "save" ? expandedOptions : []);
        showStatus(action === "restore" ? "上一版已恢复。" : "已保存，指令立即生效。", "success");
      }
    } catch (error) {
      showStatus(error.message || String(error), "error");
    } finally {
      busy = false;
      updateButtons();
    }
  }

  saveButton.addEventListener("click", () => submitProfiles("save", { profiles: readProfiles() }));
  validateButton.addEventListener("click", () => submitProfiles("validate", { profiles: readProfiles() }));
  let restoreArmed = false;
  let restoreArmTimer = null;
  function disarmRestore() {
    restoreArmed = false;
    if (restoreArmTimer) {
      clearTimeout(restoreArmTimer);
      restoreArmTimer = null;
    }
    restoreButton.textContent = "恢复上一版";
  }
  restoreButton.addEventListener("click", () => {
    if (busy) return;
    if (!restoreArmed) {
      restoreArmed = true;
      restoreButton.textContent = "再次点击确认恢复";
      showStatus("恢复上一版会替换当前配置且丢失未保存修改，请在 6 秒内再次点击“恢复上一版”确认。", "warning");
      restoreArmTimer = setTimeout(disarmRestore, 6000);
      return;
    }
    disarmRestore();
    submitProfiles("restore", {});
  });

  function healthCard(title, detail, healthy) {
    const card = element("div", "health-item");
    card.dataset.state = healthy ? "ok" : "warning";
    card.append(element("h3", "", title), element("p", "", detail));
    return card;
  }

  function diagnosticTable(title, columns, values, className) {
    const section = element("section", "diagnostic-content");
    const wrap = element("div", "table-wrap");
    const table = element("table", className);
    const labels = table.createTHead().insertRow();
    for (const column of columns) {
      const label = element("th", "", column);
      label.scope = "col";
      labels.append(label);
    }
    const body = table.createTBody();
    for (const valuesRow of values) {
      const row = body.insertRow();
      for (const value of valuesRow) row.insertCell().textContent = value;
    }
    wrap.append(table);
    section.append(element("h3", "", title), wrap);
    return section;
  }

  function renderDiagnostics(result) {
    const health = element("div", "health-grid");
    health.append(
      healthCard("ComfyUI", result.comfyui_online ? "可达" : "健康检查未通过", result.comfyui_online),
      healthCard("GPT Image", (result.gpt_image_available ? "支持图片的提供商已就绪" : "未找到支持图片的提供商")
        + (result.fallback_enabled ? " · 回退已开启" : " · 回退已关闭"), result.gpt_image_available),
    );
    diagnostics.replaceChildren(health);
    const workflows = (result.workflows || []).map((workflow) => {
      if (!workflow.loaded) return [workflow.name, workflow.enabled ? "工作流未加载" : "未启用"];
      const messages = [];
      if (workflow.missing?.length) messages.push("缺少节点 " + workflow.missing.join("、"));
      if (workflow.unknown) messages.push(workflow.unknown + " 个节点未确认");
      return [workflow.name, messages.join("；") || "已检测节点齐全"];
    });
    if (workflows.length) {
      diagnostics.append(diagnosticTable("工作流", ["工作流", "节点检查"], workflows));
    }
    const seconds = (value) => value == null ? "—" : value + " s";
    const tasks = (result.recent || []).map((task) => [
      task.id, task.backend, task.status, ...[task.queue, task.generation, task.send].map(seconds),
    ]);
    if (tasks.length) {
      diagnostics.append(diagnosticTable("最近图片任务", ["任务编号", "后端", "状态", "排队", "处理", "发送"], tasks, "task-table"));
    } else {
      diagnostics.append(element("p", "hint", "暂无近期图片任务。"));
    }
  }

  diagnoseButton.addEventListener("click", async () => {
    diagnoseButton.disabled = true;
    diagnostics.textContent = "正在检查…";
    try {
      renderDiagnostics(await page.apiGet("edit_options/diagnostics"));
      diagnoseButton.textContent = "重新检查";
    } catch (error) {
      diagnostics.textContent = "诊断失败：" + (error.message || String(error));
    } finally {
      diagnoseButton.disabled = false;
    }
  });

  updateButtons();
  load();
})();
