/* 3D drawing + user interaction. All thinking happens in Python (app/);
   this file only draws state and forwards user actions over the bridge. */

import { ForceGraph3D, SpriteText } from './vendor/graph-bundle.mjs';

const $ = (id) => document.getElementById(id);
const status = (msg) => { $('status').textContent = msg || ''; };

const COLORS = {
  node: '#7ea8d8',
  container: '#c792ea',      // overview: this node has members — zoomable
  selectedA: '#ffb454',
  selectedB: '#ff7eb6',
  label: '#dbe2ee',
  shared: '#ffffff',         // focus view: node in several focused containers
  ghost: '#4a5568',          // focus view: outside node reaching in
  ghostLabel: '#6b7687',
  boundaryLink: '#39445a',
};
// One hue per focused container, by its position in the focus list.
const FOCUS_HUES = ['#5aa9ff', '#6fd08c', '#ffb454', '#ff7eb6'];

let api;                 // window.pywebview.api
let state = { nodes: [], edges: [], llm_available: null };
let selected = [];       // up to two concept names, in click order
let layoutMode = 'manual';
let relateMode = 'ai';        // 'ai' = model tidies the sentence; 'exact' = store verbatim
let relateDirection = 'ab';   // 'ab' | 'ba' | 'both' — only used in 'exact' mode
let graph;
let focusStack = [];     // each entry: the list of focused container names
let focusData = null;    // last payload from get_focus_view
let containers = new Set();  // names that have members (derived from parents)

const inFocus = () => focusStack.length > 0;
const currentFocus = () => focusStack[focusStack.length - 1];
const focusHue = (name) => {
  const i = inFocus() ? currentFocus().indexOf(name) : -1;
  return i >= 0 ? FOCUS_HUES[i % FOCUS_HUES.length] : COLORS.node;
};

/* ---------- tiny DOM helper (never innerHTML with user text) ---------- */

// For strings handed to the 3D library's tooltip, which renders HTML.
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

/* ---------- 3D view ---------- */

function initGraph() {
  graph = new ForceGraph3D($('graph'))
    .backgroundColor('#10141c')
    .nodeColor((n) => {
      const i = selected.indexOf(n.id);
      if (i === 0) return COLORS.selectedA;
      if (i === 1) return COLORS.selectedB;
      if (inFocus()) {
        if (n.ghost) return COLORS.ghost;
        if (n.signature?.length > 1) return COLORS.shared;
        return focusHue(n.signature?.[0]);
      }
      return containers.has(n.id) ? COLORS.container : COLORS.node;
    })
    .nodeVal((n) => {
      if (n.ghost) return 1;
      const zoomable = inFocus() ? n.has_members : containers.has(n.id);
      return zoomable ? 5 : 2;
    })
    .nodeThreeObject((n) => {
      const sprite = new SpriteText(
        n.id, n.ghost ? 3 : 4, n.ghost ? COLORS.ghostLabel : COLORS.label);
      sprite.position.set(0, 8, 0);
      sprite.material.depthWrite = false;
      return sprite;
    })
    .nodeThreeObjectExtend(true)
    .nodeLabel((n) => (n.desc ? escapeHtml(n.desc) : null))
    .linkColor((l) => (l.boundary ? COLORS.boundaryLink : null))
    .linkLabel((l) => `${l.predicate} ${l.directed ? '→' : '↔'}`)
    .linkDirectionalArrowLength((l) => (l.directed ? 4 : 0))
    .linkDirectionalArrowRelPos(0.55)
    .linkCurvature((l) => l.curvature)
    .linkCurveRotation((l) => l.curveRotation)
    .linkOpacity(0.45)
    .linkWidth(0.8)
    .onNodeClick(handleNodeClick)
    .onBackgroundClick(() => { selected = []; refreshSelectionUI(); repaint(); })
    .onNodeDragEnd((n) => {
      if (inFocus()) return;  // focus view never touches saved positions
      n.fx = n.x; n.fy = n.y; n.fz = n.z;
      const entry = state.nodes.find((s) => s.name === n.id);
      if (entry) entry.position = [n.x, n.y, n.z];
      api.set_position(n.id, n.x, n.y, n.z);
    });
  window.addEventListener('resize', () =>
    graph.width($('graph').clientWidth).height($('graph').clientHeight));
  window.__graph = graph;  // debug handle, like __mainLoaded/__reportError
}

function repaint() {
  // Re-setting the accessor makes the library re-evaluate node colours.
  graph.nodeColor(graph.nodeColor());
}

// Spread parallel edges between the same pair along different curves.
function buildLinks(edges, isGhost = () => false) {
  const pairCount = new Map();
  const pairKey = (e) => [e.source, e.target].sort().join('\0');
  for (const e of edges) {
    pairCount.set(pairKey(e), (pairCount.get(pairKey(e)) ?? 0) + 1);
  }
  const pairSeen = new Map();
  return edges.map((e) => {
    const key = pairKey(e);
    const i = pairSeen.get(key) ?? 0;
    pairSeen.set(key, i + 1);
    const total = pairCount.get(key);
    return {
      source: e.source,
      target: e.target,
      predicate: e.predicate,
      directed: e.directed,
      boundary: isGhost(e.source) || isGhost(e.target),
      curvature: total > 1 ? 0.25 : 0,
      curveRotation: total > 1 ? (2 * Math.PI * i) / total : 0,
    };
  });
}

function render() {
  // While focused, any state change may alter members/ghosts/edges, so
  // re-fetch the view instead of redrawing stale focus data.
  if (inFocus()) { refreshFocus(); return; }
  containers = new Set(state.nodes.flatMap((n) => n.parents ?? []));
  const prev = new Map(graph.graphData().nodes.map((n) => [n.id, n]));
  const nodes = state.nodes.map((n) => {
    const node = prev.get(n.name) ?? { id: n.name };
    node.saved = n.position;   // manual (x,y,z) or null
    node.desc = n.description; // shown as the hover tooltip
    node.ghost = false;
    node.signature = [];
    node.has_members = containers.has(n.name);
    return node;
  });
  graph.d3Force('venn', null);
  graph.graphData({ nodes, links: buildLinks(state.edges) });
  applyLayout();
  refreshFocusUI();
  refreshSelectionUI();
}

async function applyLayout() {
  const { nodes } = graph.graphData();
  if (layoutMode === 'meaning') {
    graph.enableNodeDrag(false);
    $('layout-hint').textContent =
      'Positions are computed from meaning — similar concepts sit together. Dragging is off.';
    if (nodes.length) {
      status('computing meaning layout…');
      try {
        const pos = await api.get_meaning_positions();
        for (const n of nodes) {
          const p = pos[n.id];
          if (p) [n.fx, n.fy, n.fz] = p;
        }
        graph.d3ReheatSimulation();
      } finally { status(''); }
    }
  } else {
    graph.enableNodeDrag(true);
    $('layout-hint').textContent = 'Drag nodes to arrange them; positions are saved.';
    for (const n of nodes) {
      if (n.saved) { [n.fx, n.fy, n.fz] = n.saved; }
      else { n.fx = n.fy = n.fz = undefined; }
    }
    graph.d3ReheatSimulation();
  }
}

/* ---------- focus view (zoom into containers) ---------- */

async function focusOn(focused) {
  try {
    focusData = await api.get_focus_view(focused);
  } catch (err) {
    status(`couldn’t focus: ${err}`);
    return;
  }
  focusStack.push(focusData.focused);
  selected = [];
  renderFocus();
}

async function refreshFocus() {
  // Re-fetch the current focus after a mutation (membership added/removed).
  try {
    focusData = await api.get_focus_view(currentFocus());
  } catch (err) {
    // e.g. a focused container was deleted — drop back to the overview.
    focusStack = [];
    focusData = null;
    render();
    return;
  }
  focusStack[focusStack.length - 1] = focusData.focused;
  renderFocus();
}

function focusBack(toDepth = focusStack.length - 1) {
  focusStack = focusStack.slice(0, toDepth);
  selected = [];
  if (!inFocus()) { focusData = null; render(); return; }
  focusOnCurrent();
}

async function focusOnCurrent() {
  focusData = await api.get_focus_view(currentFocus());
  renderFocus();
}

// Anchor per non-empty signature group: single-container groups sit at the
// vertices of a regular polygon; an intersection group sits at the centroid
// of its containers' anchors. Nodes are then gently pulled toward their
// group's anchor on top of the normal force layout (Venn-style clustering).
function groupAnchors() {
  const focused = focusData.focused;
  const R = 150;
  const single = new Map();
  focused.forEach((name, i) => {
    if (focused.length === 1) { single.set(name, [0, 0, 0]); return; }
    const angle = (2 * Math.PI * i) / focused.length;
    single.set(name, [R * Math.cos(angle), R * Math.sin(angle), 0]);
  });
  const anchors = new Map();
  for (const g of focusData.groups) {
    const pts = g.signature.map((name) => single.get(name));
    anchors.set(g.signature.join('\0'), [
      pts.reduce((s, p) => s + p[0], 0) / pts.length,
      pts.reduce((s, p) => s + p[1], 0) / pts.length,
      pts.reduce((s, p) => s + p[2], 0) / pts.length,
    ]);
  }
  return anchors;
}

function vennForce() {
  const anchors = groupAnchors();
  let nodes = [];
  const STRENGTH = 0.05;
  const force = (alpha) => {
    for (const n of nodes) {
      const anchor = anchors.get((n.signature ?? []).join('\0'));
      if (!anchor) continue;  // ghosts drift free
      n.vx += (anchor[0] - n.x) * STRENGTH * alpha;
      n.vy += (anchor[1] - n.y) * STRENGTH * alpha;
      n.vz += (anchor[2] - (n.z ?? 0)) * STRENGTH * alpha;
    }
  };
  force.initialize = (ns) => { nodes = ns; };
  return force;
}

function renderFocus() {
  const ghosts = new Set(focusData.nodes.filter((n) => n.ghost).map((n) => n.name));
  const prev = new Map(graph.graphData().nodes.map((n) => [n.id, n]));
  const nodes = focusData.nodes.map((f) => {
    const node = prev.get(f.name) ?? { id: f.name };
    node.desc = f.description;
    node.ghost = f.ghost;
    node.signature = f.signature;
    node.has_members = f.has_members;
    node.saved = null;
    // Focus layout is always force-driven: saved overview positions stay
    // untouched, and nothing here is ever written back.
    node.fx = node.fy = node.fz = undefined;
    return node;
  });
  graph.enableNodeDrag(false);
  graph.d3Force('venn', vennForce());
  graph.graphData({ nodes, links: buildLinks(focusData.edges, (n) => ghosts.has(n)) });
  graph.d3ReheatSimulation();
  refreshFocusUI();
  refreshSelectionUI();
}

function refreshFocusUI() {
  $('focus-section').hidden = !inFocus();
  $('layout-section').hidden = inFocus();
  const crumbs = $('focus-crumbs');
  const legend = $('focus-legend');
  crumbs.replaceChildren();
  legend.replaceChildren();
  if (!inFocus()) return;

  crumbs.append(el('span', {
    class: 'sel-chip', onclick: () => focusBack(0),
  }, 'Overview'));
  focusStack.forEach((focused, depth) => {
    crumbs.append(' › ');
    crumbs.append(el('span', {
      class: depth === focusStack.length - 1 ? 'sel-chip current' : 'sel-chip',
      onclick: () => focusBack(depth + 1),
    }, focused.join(' + ')));
  });

  for (const g of focusData.groups) {
    const color = g.signature.length > 1 ? COLORS.shared : focusHue(g.signature[0]);
    legend.append(el('div', { class: 'legend-row' },
      el('span', { class: 'dot', style: `background:${color}` }),
      `${g.signature.join(' ∩ ')} — ${g.count}`));
  }
  if (!focusData.groups.length) {
    legend.append(el('p', { class: 'hint' }, 'This container has no members yet.'));
  }
}

/* ---------- selection ---------- */

let lastClick = { name: null, at: 0 };

function handleNodeClick(node) {
  // Manual double-click detection: two clicks on the same node within 350 ms
  // zoom into it (if it has members). The second toggleSelect undoes the
  // first, so selection is left as it was before the double-click.
  const now = Date.now();
  const isDouble = node.id === lastClick.name && now - lastClick.at < 350;
  lastClick = { name: node.id, at: now };
  toggleSelect(node);
  if (isDouble && node.has_members) focusOn([node.id]);
}

function toggleSelect(node) {
  const name = node.id;
  if (selected.includes(name)) selected = selected.filter((s) => s !== name);
  else selected = [...selected, name].slice(-2);
  refreshSelectionUI();
  repaint();
}

function refreshSelectionUI() {
  selected = selected.filter((name) => state.nodes.some((n) => n.name === name));
  const chips = $('selection-chips');
  const details = $('selection-details');
  const actions = $('selection-actions');
  chips.replaceChildren();
  details.replaceChildren();
  actions.replaceChildren();

  if (!selected.length) {
    chips.textContent = 'Click a node to select it (up to two).';
  } else {
    for (const name of selected) chips.append(el('span', { class: 'sel-chip' }, name));
    for (const name of selected) details.append(descriptionEditor(name));
    for (const name of selected) {
      for (const btn of membershipActions(name)) actions.append(btn);
      actions.append(el('button', {
        class: 'small danger',
        onclick: () => removeConcept(name),
      }, `Delete “${name}”`));
    }
    if (selected.length === 2 && !inFocus()) {
      const [a, b] = selected;
      actions.append(
        el('button', { class: 'small', onclick: () => addMember(a, b) },
          `“${a}” is part of “${b}”`),
        el('button', { class: 'small', onclick: () => addMember(b, a) },
          `“${b}” is part of “${a}”`));
    }
  }

  $('relate-section').hidden = selected.length !== 2;
  $('between-section').hidden = selected.length !== 2;
  if (selected.length === 2) {
    renderRelateMode();
    renderBetween();
  }
}

// Membership quick actions for one selected node — focus/zoom on containers,
// and, inside a focus, adding ghosts to / removing members from the focused
// containers.
function membershipActions(name) {
  const buttons = [];
  if (!inFocus()) {
    if (containers.has(name)) {
      buttons.push(el('button', {
        class: 'small primary', onclick: () => focusOn([name]),
      }, `Focus on “${name}”`));
    }
    return buttons;
  }
  const fNode = focusData.nodes.find((n) => n.name === name);
  if (!fNode) return buttons;
  if (fNode.has_members) {
    buttons.push(el('button', {
      class: 'small primary', onclick: () => focusOn([name]),
    }, `Zoom into “${name}”`));
    if (!currentFocus().includes(name)) {
      buttons.push(el('button', {
        class: 'small', onclick: () => focusOn([...currentFocus(), name]),
      }, 'Add to focus'));
    }
  }
  for (const c of currentFocus()) {
    if (fNode.ghost || !fNode.signature.includes(c)) {
      buttons.push(el('button', {
        class: 'small', onclick: () => addMember(name, c),
      }, `Add to “${c}”`));
    } else {
      buttons.push(el('button', {
        class: 'small danger', onclick: () => removeMember(name, c),
      }, `Remove from “${c}”`));
    }
  }
  return buttons;
}

async function addMember(child, parent) {
  try {
    const resp = await api.add_member(child, parent);
    state = resp;
    status(resp.added ? `“${child}” is now part of “${parent}”`
                      : `“${child}” is already part of “${parent}”`);
  } catch (err) {
    status(`couldn’t add: ${err}`);  // e.g. a membership cycle, named
    return;
  }
  render();
}

async function removeMember(child, parent) {
  try {
    state = await api.remove_member(child, parent);
    status(`“${child}” removed from “${parent}”`);
  } catch (err) {
    status(`couldn’t remove: ${err}`);
    return;
  }
  render();
}

function descriptionEditor(name) {
  const current = state.nodes.find((n) => n.name === name)?.description ?? '';
  const input = el('textarea', { class: 'desc', placeholder: 'Definition (optional)…' });
  input.value = current;
  const save = el('button', {
    class: 'small',
    onclick: async () => {
      save.disabled = true;
      try {
        state = await api.set_description(name, input.value.trim());
        status(`saved the definition of “${name}”`);
        render();
      } catch (err) {
        status(`couldn’t save definition: ${err}`);
        save.disabled = false;
      }
    },
  }, 'Save definition');
  return el('div', { class: 'card' },
    el('div', { class: 'desc-label' }, name),
    input, save);
}

/* ---------- relationship input mode ---------- */

function setRelateMode(mode) {
  relateMode = mode;
  renderRelateMode();
}

function renderRelateMode() {
  $('relate-mode-ai').classList.toggle('active', relateMode === 'ai');
  $('relate-mode-exact').classList.toggle('active', relateMode === 'exact');
  $('relate-hint').textContent = relateMode === 'ai'
    ? 'Write a full sentence; the local model turns it into a tidy edge.'
    : 'Your words become the edge exactly as typed. Pick the direction below.';
  $('relate-input').placeholder = relateMode === 'ai'
    ? 'In plain English — e.g. “the determinant being zero is what makes this an eigenvalue”'
    : 'The relationship wording — e.g. “is a special case of”';
  renderRelateDirection();
}

function renderRelateDirection() {
  const box = $('relate-direction');
  box.hidden = relateMode !== 'exact' || selected.length !== 2;
  if (box.hidden) return;
  const [a, b] = selected;
  const dirBtn = (value, label) => el('button', {
    class: relateDirection === value ? 'active' : '',
    title: label,
    onclick: () => { relateDirection = value; renderRelateDirection(); },
  }, label);
  box.replaceChildren(
    dirBtn('ab', `${a} → ${b}`),
    dirBtn('ba', `${b} → ${a}`),
    dirBtn('both', `${a} ↔ ${b}`),
  );
}

function renderBetween() {
  const [a, b] = selected;
  const box = $('between-edges');
  box.replaceChildren();
  const between = state.edges.filter(
    (e) => (e.source === a && e.target === b) || (e.source === b && e.target === a));
  if (!between.length) {
    box.append(el('p', { class: 'hint' }, 'No relationship between these two yet.'));
  }
  for (const e of between) {
    box.append(el('div', { class: 'card' },
      el('div', {}, `${e.source} `, el('span', { class: 'pred' }, e.predicate),
        ` ${e.directed ? '→' : '↔'} ${e.target}`),
      el('button', {
        class: 'small danger',
        onclick: async () => {
          state = await api.remove_relationship(e.source, e.target, e.predicate);
          render();
        },
      }, 'Remove')));
  }
  $('explain-out').replaceChildren();
}

/* ---------- actions ---------- */

async function addConcept() {
  const input = $('concept-input');
  const desc = $('concept-desc');
  const name = input.value.trim();
  if (!name) return;
  const resp = await api.add_concept(name, desc.value.trim());
  state = resp;
  input.value = '';
  if (resp.created) desc.value = '';
  else status(`“${name}” is already on the map`);
  render();
}

async function removeConcept(name) {
  const touching = state.edges.filter((e) => e.source === name || e.target === name);
  if (touching.length &&
      !confirm(`Delete “${name}” and its ${touching.length} relationship(s)?`)) return;
  state = await api.remove_concept(name);
  render();
}

async function addRelationship() {
  const [a, b] = selected;
  const sentence = $('relate-input').value.trim();
  if (!sentence) return;
  const btn = $('relate-add');
  btn.disabled = true;
  try {
    let resp, e;
    if (relateMode === 'exact') {
      status('storing…');
      const directed = relateDirection !== 'both';
      const [source, target] = relateDirection === 'ba' ? [b, a] : [a, b];
      resp = await api.add_edge_direct(source, target, sentence, directed);
      e = { source, target, predicate: sentence, directed };
    } else {
      status(state.llm_available ? 'asking the local model to tidy that up…' : 'storing…');
      resp = await api.add_relationship(a, b, sentence);
      e = resp.edge;
    }
    state = resp;
    $('relate-input').value = '';
    const note = resp.added
      ? `Stored: ${e.source} — ${e.predicate} ${e.directed ? '→' : '↔'} ${e.target}`
      : 'That exact relationship is already on the map.';
    $('relate-result').textContent = note;
    render();
  } catch (err) {
    $('relate-result').textContent = `Couldn’t add that: ${err}`;
  } finally {
    btn.disabled = false;
    status('');
  }
}

async function suggest() {
  const out = $('suggest-out');
  out.replaceChildren(el('p', { class: 'spin' }, 'scoring unconnected pairs…'));
  const suggestions = await api.suggest(8);
  out.replaceChildren();
  if (!suggestions.length) {
    out.append(el('p', { class: 'hint' },
      'Nothing to suggest — add more concepts, or everything is already connected.'));
    return;
  }
  for (const s of suggestions) {
    const card = el('div', { class: 'card' });
    card.append(
      el('div', {}, `${s.a} ↔ ${s.b}`),
      el('div', { class: 'score' },
        `likely related — score ${(s.score * 100).toFixed(0)}%` +
        ` (meaning ${(s.meaning * 100).toFixed(0)}%, structure ${(s.structure * 100).toFixed(0)}%)`));
    const actions = el('div', {});
    if (state.llm_available) {
      actions.append(el('button', {
        class: 'small',
        onclick: async (ev) => {
          ev.target.disabled = true;
          ev.target.textContent = 'wording…';
          try {
            const e = await api.verbalise(s.a, s.b);
            actions.replaceChildren(
              el('div', { class: 'facet' }, `${e.source} `,
                el('span', { class: 'pred' }, e.predicate),
                ` ${e.directed ? '→' : '↔'} ${e.target}`),
              el('button', {
                class: 'small primary',
                onclick: async () => {
                  state = await api.add_edge_direct(e.source, e.target, e.predicate, e.directed);
                  card.remove();
                  render();
                },
              }, 'Add to map'),
              el('button', { class: 'small', onclick: () => card.remove() }, 'Dismiss'));
          } catch (err) {
            ev.target.textContent = `model error: ${err}`;
          }
        },
      }, 'Suggest wording'));
    }
    actions.append(
      el('button', {
        class: 'small',
        onclick: () => { selected = [s.a, s.b]; refreshSelectionUI(); repaint(); },
      }, 'Select pair'),
      el('button', { class: 'small', onclick: () => card.remove() }, 'Dismiss'));
    card.append(actions);
    out.append(card);
  }
}

async function suggestMembers() {
  const out = $('member-suggest-out');
  out.replaceChildren(el('p', { class: 'spin' }, 'scoring candidate members…'));
  const suggestions = await api.suggest_members(8);
  out.replaceChildren();
  if (!suggestions.length) {
    out.append(el('p', { class: 'hint' },
      'Nothing to suggest — mark a concept as part of another first ' +
      '(select two nodes), then suggestions can grow those containers.'));
    return;
  }
  for (const s of suggestions) {
    const card = el('div', { class: 'card' });
    card.append(
      el('div', {}, `${s.node} ⊂ ${s.container}`),
      el('div', { class: 'score' },
        `likely a member — score ${(s.score * 100).toFixed(0)}%` +
        ` (meaning ${(s.meaning * 100).toFixed(0)}%, structure ${(s.structure * 100).toFixed(0)}%)`),
      el('div', {},
        el('button', {
          class: 'small primary',
          onclick: async () => { await addMember(s.node, s.container); card.remove(); },
        }, 'Add to map'),
        el('button', { class: 'small', onclick: () => card.remove() }, 'Dismiss')));
    out.append(card);
  }
}

async function explain() {
  const [a, b] = selected;
  const out = $('explain-out');
  out.replaceChildren(el('p', { class: 'spin' }, 'tracing paths…'));
  const result = await api.explain(a, b);
  out.replaceChildren();
  if (!result.paths.length) {
    out.append(el('p', { class: 'hint' },
      'No chain of relationships connects these two yet.'));
    return;
  }
  const heading = result.agree
    ? 'The paths agree — one way to see it:'
    : `The paths disagree — ${result.groups.length} distinct facets:`;
  out.append(el('p', { class: 'hint' }, heading));
  if (result.facet_sentences) {
    result.facet_sentences.forEach((sentence) =>
      out.append(el('div', { class: 'card facet' }, sentence)));
  } else {
    out.append(el('p', { class: 'hint' },
      'Local model is offline — showing the raw chains only.'));
  }
  for (const text of result.path_texts) {
    out.append(el('div', { class: 'chain' }, text));
  }
  if (result.used_reverse) {
    out.append(el('p', { class: 'hint' },
      'No chain follows every arrow forward; steps written ← read against the arrow.'));
  }
}

async function linchpins() {
  const out = $('linchpin-out');
  out.replaceChildren(el('p', { class: 'spin' }, 'measuring…'));
  const items = await api.linchpins(5);
  out.replaceChildren();
  if (!items.length) {
    out.append(el('p', { class: 'hint' },
      'No linchpins yet — they appear once chains of reasoning share edges.'));
    return;
  }
  for (const { edge, score } of items) {
    out.append(el('div', { class: 'card' },
      el('div', {}, `${edge.source} `, el('span', { class: 'pred' }, edge.predicate),
        ` ${edge.directed ? '→' : '↔'} ${edge.target}`),
      el('div', { class: 'score' },
        `${(score * 100).toFixed(0)}% of shortest chains pass through this`)));
  }
}

/* ---------- snapshots ---------- */

async function refreshSnapshots(names) {
  const list = names ?? await api.list_snapshots();
  const select = $('snap-list');
  select.replaceChildren();
  for (const name of list) select.append(el('option', { value: name }, name));
}

async function saveSnapshot() {
  const { snapshot, snapshots } = await api.save_snapshot();
  status(snapshot ? `saved ${snapshot}` : 'nothing changed since the last snapshot');
  refreshSnapshots(snapshots);
}

async function restoreSnapshot() {
  const name = $('snap-list').value;
  if (!name) return;
  if (!confirm(`Roll the map back to ${name}? (Your current map is snapshotted first.)`)) return;
  state = await api.restore_snapshot(name);
  render();
  refreshSnapshots();
}

/* ---------- LLM badge ---------- */

async function pollLLM() {
  const badge = $('llm-badge');
  const val = await api.llm_status();
  state.llm_available = val;
  if (val === null) { setTimeout(pollLLM, 800); return; }
  badge.classList.add(val ? 'on' : 'off');
  $('llm-text').textContent = val
    ? `local model ready (${state.model ?? 'ollama'})`
    : 'local model offline — sentences stored verbatim';
}

/* ---------- wiring ---------- */

function bindPanel() {
  $('add-concept').addEventListener('click', addConcept);
  $('concept-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') addConcept(); });
  $('relate-add').addEventListener('click', addRelationship);
  $('relate-mode-ai').addEventListener('click', () => setRelateMode('ai'));
  $('relate-mode-exact').addEventListener('click', () => setRelateMode('exact'));
  $('suggest-btn').addEventListener('click', suggest);
  $('member-suggest-btn').addEventListener('click', suggestMembers);
  $('focus-back').addEventListener('click', () => focusBack());
  $('explain-btn').addEventListener('click', explain);
  $('linchpin-btn').addEventListener('click', linchpins);
  $('snap-btn').addEventListener('click', saveSnapshot);
  $('snap-restore').addEventListener('click', restoreSnapshot);
  $('mode-manual').addEventListener('click', () => setMode('manual'));
  $('mode-meaning').addEventListener('click', () => setMode('meaning'));
}

function setMode(mode) {
  layoutMode = mode;
  $('mode-manual').classList.toggle('active', mode === 'manual');
  $('mode-meaning').classList.toggle('active', mode === 'meaning');
  applyLayout();
}

async function start() {
  api = window.pywebview.api;
  bindPanel();
  try {
    initGraph();
  } catch (err) {
    // 3D view failed (usually WebGL); keep the panel alive regardless.
    window.__reportError?.(`3D view failed: ${err.message ?? err}`);
    graph = null;
  }
  state = await api.get_state();
  if (graph) render(); else refreshSelectionUI();
  refreshSnapshots();
  pollLLM();
}

window.__mainLoaded = true;
// A deferred module can finish loading after pywebview has already announced
// itself — handle both orders.
if (window.pywebview?.api) start();
else window.addEventListener('pywebviewready', start);
