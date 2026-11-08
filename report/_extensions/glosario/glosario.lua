-- Shortcodes del glosario del sitio.
--
--   {{< term id >}}            enlaza el término `id` al glosario, con su nombre como texto
--   {{< term id "texto" >}}    igual, con otro texto visible (plural, minúsculas...)
--   {{< glosario-indice >}}    índice alfabético de todos los términos (página del glosario)
--
-- Única fuente: report/glosario.qmd. Cada término es un encabezado `### Nombre {#id}` y su
-- primer párrafo es la definición corta que se muestra al pasar el cursor (_glosario.html).

local GLOSSARY_FILE = "glosario.qmd"
local GLOSSARY_PAGE = "glosario.html"

local entries = nil -- id → { title, definition }
local order = {}    -- ids en orden de aparición

local function load_entries()
  if entries then
    return entries
  end
  entries = {}
  local path = pandoc.path.join({ quarto.project.directory, GLOSSARY_FILE })
  local handle = io.open(path, "r")
  if not handle then
    quarto.log.error("glosario: no se encontró " .. path)
    return entries
  end
  local doc = pandoc.read(handle:read("a"), "markdown")
  handle:close()

  local current = nil
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" then
      current = nil
      if block.level == 3 and block.identifier ~= "" then
        current = block.identifier
        entries[current] = { title = pandoc.utils.stringify(block.content) }
        table.insert(order, current)
      end
    elseif current and block.t == "Para" then
      entries[current].definition = pandoc.utils.stringify(block.content)
      current = nil
    end
  end
  return entries
end

local function href(id)
  local offset = quarto.project.offset or "."
  return offset .. "/" .. GLOSSARY_PAGE .. "#" .. id
end

local function term(args)
  local id = pandoc.utils.stringify(args[1] or "")
  local entry = load_entries()[id]
  if entry == nil then
    quarto.log.error("glosario: término desconocido «" .. id .. "»")
    return pandoc.Strong(pandoc.Str("¿" .. id .. "?"))
  end
  local label = args[2] and pandoc.utils.stringify(args[2]) or entry.title
  if not quarto.doc.is_format("html") then
    return pandoc.Str(label)
  end
  local attributes = { ["data-term"] = entry.title, ["data-def"] = entry.definition or "" }
  return pandoc.Link(label, href(id), "", pandoc.Attr("", { "term" }, attributes))
end

local function index()
  local all = load_entries()
  local ids = { table.unpack(order) }
  table.sort(ids, function(a, b)
    return all[a].title:lower() < all[b].title:lower()
  end)
  local items = {}
  for _, id in ipairs(ids) do
    table.insert(items, { pandoc.Plain({ pandoc.Link(all[id].title, "#" .. id) }) })
  end
  return pandoc.Div({ pandoc.BulletList(items) }, pandoc.Attr("", { "glosario-indice" }))
end

return {
  ["term"] = term,
  ["glosario-indice"] = index,
}
