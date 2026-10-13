package chatmessages

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// anthropicSchemaKeys is the whole vocabulary an Anthropic tool input_schema
// accepts. Every other key is dropped, which is why a legacy `$ref` block has
// to be inlined first: the definition it points at would otherwise be removed
// and leave a dangling pointer.
var anthropicSchemaKeys = map[string]struct{}{
	"type": {}, "properties": {}, "additionalProperties": {}, "required": {},
	"$defs": {}, "strict": {},
}

// sanitizeInputSchemaForAnthropic coerces an arbitrary tool schema into the
// shape the Messages dialect accepts: an object schema, draft-04 `definitions`
// and OpenAPI `components.schemas` refs inlined, and every key outside the
// accepted set removed. The schema is copied before any of this — the caller's
// map is never written through, which is the library's pure-function
// contract.
func sanitizeInputSchemaForAnthropic(schema any) (map[string]any, error) {
	out, _ := jsonx.AsMap(schema)
	if out == nil {
		out = map[string]any{}
	} else {
		out = deepCopy(out).(map[string]any)
	}
	if jsonx.GetString(out, "type") != "object" {
		out["type"] = "object"
	}
	if _, ok := out["properties"]; !ok {
		out["properties"] = map[string]any{}
	}
	if err := unpackLegacyDefs(out); err != nil {
		return nil, err
	}
	for key := range out {
		if _, ok := anthropicSchemaKeys[key]; !ok {
			delete(out, key)
		}
	}
	return out, nil
}

// defsMaxInlinedBytes caps the cumulative JSON size of every inlined target:
// request-supplied schemas cannot expand into a schema-bomb before reaching
// the upstream, and an expansion past the budget fails the request rather
// than materialising.
const defsMaxInlinedBytes = 10_000_000

// unpackLegacyDefs inlines `$ref`s backed by draft-04 `definitions` or OpenAPI
// `components.schemas` into the schema itself, then drops those blocks. `$defs`
// survives as a ref target and joins the resolution scope — on a name clash the
// `$defs` body wins, as the reference resolver reads it — and definitions a
// resolved target carries resolve against the scope it is resolved in. A
// circular reference is left in place: it names a definition the schema can
// never fully spell out, and the target is the one to judge it.
func unpackLegacyDefs(schema map[string]any) error {
	_, hasDefinitions := schema["definitions"]
	components, hasComponents := jsonx.AsMap(schema["components"])
	_, hasSchemas := components["schemas"]
	if !hasDefinitions && !(hasComponents && hasSchemas) {
		return nil
	}
	// `definitions` shadows `components.schemas` — it is the JSON-Schema-native
	// namespace, and one name can resolve to only one body.
	defs := map[string]any{}
	if schemas, ok := jsonx.AsMap(components["schemas"]); ok {
		for k, v := range schemas {
			defs[k] = v
		}
	}
	if definitions, ok := jsonx.AsMap(schema["definitions"]); ok {
		for k, v := range definitions {
			defs[k] = v
		}
	}
	delete(schema, "components")
	delete(schema, "definitions")
	return unpackSchemaRefs(schema, defs)
}

// unpackSchemaRefs walks the whole schema tree and replaces each {"$ref": ...}
// whose last path segment names a definition in scope with a copy of it. The
// scope starts from the caller's defs plus the schema's own `$defs` (the
// native namespace shadows legacy ones) and widens with the `$defs` and
// `definitions` each node and each resolved target carry; a reference already
// being resolved on the current path is circular and stays as written; and
// each inlined target counts against the byte budget before it is copied.
func unpackSchemaRefs(schema map[string]any, defs map[string]any) error {
	rootDefs := map[string]any{}
	for k, v := range defs {
		rootDefs[k] = v
	}
	if d, ok := jsonx.AsMap(schema["$defs"]); ok {
		for k, v := range d {
			rootDefs[k] = v
		}
	}
	if d, ok := jsonx.AsMap(schema["definitions"]); ok {
		for k, v := range d {
			rootDefs[k] = v
		}
	}
	type slot struct {
		node   any
		parent any
		key    any
		defs   map[string]any
		chain  map[string]struct{}
	}
	inlined := 0
	queue := []slot{{node: schema, defs: rootDefs, chain: map[string]struct{}{}}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		switch node := cur.node.(type) {
		case map[string]any:
			if ref := jsonx.GetString(node, "$ref"); ref != "" {
				name := lastPathSegment(ref)
				if _, circular := cur.chain[name]; circular {
					continue
				}
				target, known := cur.defs[name]
				if !known || target == nil {
					continue
				}
				if b, err := jsonx.Marshal(target); err == nil {
					inlined += len(b)
					if inlined > defsMaxInlinedBytes {
						return fmt.Errorf("inlined schema exceeded the %d-byte budget: refusing to expand %q further", defsMaxInlinedBytes, name)
					}
				}
				childDefs := map[string]any{}
				for k, v := range cur.defs {
					childDefs[k] = v
				}
				if tm, ok := target.(map[string]any); ok {
					if d, ok := jsonx.AsMap(tm["$defs"]); ok {
						for k, v := range d {
							childDefs[k] = v
						}
					}
					if d, ok := jsonx.AsMap(tm["definitions"]); ok {
						for k, v := range d {
							childDefs[k] = v
						}
					}
				}
				resolved := deepCopy(target)
				replaceSlot(cur.parent, cur.key, resolved, schema)
				chain := make(map[string]struct{}, len(cur.chain)+1)
				for k := range cur.chain {
					chain[k] = struct{}{}
				}
				chain[name] = struct{}{}
				queue = append(queue, slot{node: resolved, parent: cur.parent, key: cur.key, defs: childDefs, chain: chain})
				continue
			}
			currentDefs := cur.defs
			nd, hasD := jsonx.AsMap(node["$defs"])
			nl, hasL := jsonx.AsMap(node["definitions"])
			if hasD || hasL {
				currentDefs = make(map[string]any, len(cur.defs)+len(nd)+len(nl))
				for k, v := range cur.defs {
					currentDefs[k] = v
				}
				for k, v := range nd {
					currentDefs[k] = v
				}
				for k, v := range nl {
					currentDefs[k] = v
				}
			}
			for k, v := range node {
				queue = append(queue, slot{node: v, parent: node, key: k, defs: currentDefs, chain: cur.chain})
			}
		case []any:
			for i, v := range node {
				queue = append(queue, slot{node: v, parent: node, key: i, defs: cur.defs, chain: cur.chain})
			}
		}
	}
	return nil
}

// replaceSlot writes a resolved ref back through the container that held it.
func replaceSlot(parent any, key any, resolved any, root map[string]any) {
	switch p := parent.(type) {
	case map[string]any:
		if k, ok := key.(string); ok {
			p[k] = resolved
		}
	case []any:
		if i, ok := key.(int); ok && i < len(p) {
			p[i] = resolved
		}
	case nil:
		// The schema itself was a ref: it becomes the resolved target.
		for k := range root {
			delete(root, k)
		}
		if m, ok := jsonx.AsMap(resolved); ok {
			for k, v := range m {
				root[k] = v
			}
		}
	}
}

func lastPathSegment(ref string) string {
	for i := len(ref) - 1; i >= 0; i-- {
		if ref[i] == '/' {
			return ref[i+1:]
		}
	}
	return ref
}

// outputSchemaConstraints maps each JSON Schema keyword Anthropic's structured
// output rejects onto the note that replaces it. Anthropic compiles the schema
// into a constrained-decoding grammar and refuses a schema carrying these, so
// the constraint is dropped and described in the schema's own description
// instead of silently losing what the caller asked for.
var outputSchemaConstraints = []struct{ key, note string }{
	{"minItems", "minimum number of items: %v"},
	{"maxItems", "maximum number of items: %v"},
	{"uniqueItems", "all array items must be unique"},
	{"contains", "array must contain an item matching: %v"},
	{"minContains", "minimum number of matching items: %v"},
	{"maxContains", "maximum number of matching items: %v"},
	{"prefixItems", "leading items must match, in order: %v"},
	{"minimum", "minimum value: %v"},
	{"maximum", "maximum value: %v"},
	{"exclusiveMinimum", "exclusive minimum value: %v"},
	{"exclusiveMaximum", "exclusive maximum value: %v"},
	{"multipleOf", "must be a multiple of %v"},
	{"minLength", "minimum length: %v"},
	{"maxLength", "maximum length: %v"},
	{"minProperties", "minimum number of properties: %v"},
	{"maxProperties", "maximum number of properties: %v"},
	{"patternProperties", "properties whose names match each pattern must satisfy: %v"},
	{"propertyNames", "property names must satisfy: %v"},
	{"dependentRequired", "dependent required properties: %v"},
	{"dependentSchemas", "dependent schemas: %v"},
	{"unevaluatedProperties", "unevaluated properties must satisfy: %v"},
	{"if", "conditional (if): %v"},
	{"then", "conditional (then): %v"},
	{"else", "conditional (else): %v"},
	{"not", "must not match: %v"},
}

// filterOutputSchemaForAnthropic strips the JSON Schema keywords Anthropic's
// structured output cannot compile, folding each into the schema's description
// so the constraint is still stated, and rewrites `oneOf` to `anyOf` — the one
// combinator Anthropic accepts.
func filterOutputSchemaForAnthropic(schema any) any {
	m, ok := jsonx.AsMap(schema)
	if !ok {
		return schema
	}
	out := map[string]any{}
	var notes []string
	for _, c := range outputSchemaConstraints {
		value, present := m[c.key]
		if !present {
			continue
		}
		if b, isBool := value.(bool); isBool && !b {
			continue
		}
		if len(c.note) > 0 && strings.Contains(c.note, "%v") {
			notes = append(notes, fmt.Sprintf(c.note, formatConstraint(value)))
		} else {
			notes = append(notes, c.note)
		}
	}
	if len(notes) > 0 {
		note := "Note: " + strings.Join(notes, ", ") + "."
		if desc := jsonx.GetString(m, "description"); desc != "" {
			out["description"] = desc + " " + note
		} else {
			out["description"] = note
		}
	}
	dropsType := enumConflictsWithType(m)
	// The two combinator keys merge into the one the target reads in a fixed
	// order: map iteration is not ordered, and the same request must fold the
	// same way every time. oneOf rides first — it is the spelling the merge
	// exists to rewrite.
	for _, key := range []string{"oneOf", "anyOf"} {
		list, ok := jsonx.AsSlice(m[key])
		if !ok {
			continue
		}
		var filtered []any
		for _, item := range list {
			filtered = append(filtered, filterOutputSchemaForAnthropic(item))
		}
		out["anyOf"] = append(existingList(out["anyOf"]), filtered...)
	}
	for key, value := range m {
		if isOutputSchemaConstraint(key) {
			continue
		}
		if key == "type" && dropsType {
			continue
		}
		if key == "description" {
			if _, handled := out["description"]; handled {
				continue
			}
		}
		switch key {
		case "properties", "$defs":
			if sub, ok := jsonx.AsMap(value); ok {
				filtered := map[string]any{}
				for k, v := range sub {
					filtered[k] = filterOutputSchemaForAnthropic(v)
				}
				out[key] = filtered
				continue
			}
		case "items":
			if _, ok := jsonx.AsMap(value); ok {
				out[key] = filterOutputSchemaForAnthropic(value)
				continue
			}
		case "anyOf", "oneOf":
			// Handled in the fixed-order pass above.
			continue
		case "allOf":
			if list, ok := jsonx.AsSlice(value); ok {
				var filtered []any
				for _, item := range list {
					filtered = append(filtered, filterOutputSchemaForAnthropic(item))
				}
				out[key] = filtered
				continue
			}
		}
		out[key] = value
	}
	// Anthropic requires additionalProperties=false on every object schema.
	if jsonx.GetString(out, "type") == "object" {
		out["additionalProperties"] = false
	}
	return out
}

func existingList(v any) []any {
	if list, ok := jsonx.AsSlice(v); ok {
		return list
	}
	return nil
}

func isOutputSchemaConstraint(key string) bool {
	for _, c := range outputSchemaConstraints {
		if c.key == key {
			return true
		}
	}
	return false
}

// formatConstraint renders a dropped constraint's value for the advisory note,
// serializing a sub-schema so the note says what was actually required. The
// serialization spells the value the way the reference's serializer does:
// ", " and ": " separators, no HTML escaping, non-ASCII escaped, and numbers
// as the reference's float repr writes them; a boolean reads as the
// capitalised word its native spelling uses.
func formatConstraint(value any) string {
	switch t := value.(type) {
	case map[string]any, []any:
		return pythonJSONText(t)
	case json.Number:
		if strings.ContainsAny(t.String(), ".eE") {
			if f, err := t.Float64(); err == nil {
				return pythonFloatText(f)
			}
		}
		return t.String()
	case bool:
		if t {
			return "True"
		}
		return "False"
	}
	return jsonx.String(value)
}

// pythonFloatText spells a float the way the reference's repr does: shortest
// round-trip digits, an exponent for the very large and very small, and a
// trailing .0 on a whole number written as a float.
func pythonFloatText(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// pythonJSONText renders a value with the reference serializer's spelling:
// ", " and ": " separators, no HTML escaping, and non-ASCII escaped as
// \uXXXX (astral characters as surrogate pairs). Map keys are sorted — the
// pipeline reads schemas from maps, which do not carry the document's key
// order, so a single-key schema (the common case) matches exactly and a
// multi-key one is at least deterministic.
func pythonJSONText(v any) string {
	var b strings.Builder
	writePythonJSON(&b, v)
	return b.String()
}

func writePythonJSON(b *strings.Builder, v any) {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			writePythonJSONString(b, k)
			b.WriteString(": ")
			writePythonJSON(b, t[k])
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, item := range t {
			if i > 0 {
				b.WriteString(", ")
			}
			writePythonJSON(b, item)
		}
		b.WriteByte(']')
	case json.Number:
		b.WriteString(formatConstraint(t))
	case string:
		writePythonJSONString(b, t)
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	default:
		b.WriteString(jsonx.String(t))
	}
}

// writePythonJSONString quotes a string with the reference serializer's
// escapes: the two mandatory characters, the short control escapes, and
// \uXXXX for every other non-ASCII rune.
func writePythonJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20:
				fmt.Fprintf(b, `\u%04x`, r)
			case r < 0x80:
				b.WriteRune(r)
			case r <= 0xFFFF:
				fmt.Fprintf(b, `\u%04x`, r)
			default:
				r -= 0x10000
				fmt.Fprintf(b, `\u%04x\u%04x`, 0xD800+(r>>10), 0xDC00+(r&0x3FF))
			}
		}
	}
	b.WriteByte('"')
}

// enumConflictsWithType reports whether a schema's enum cannot match its
// declared type, in which case the declared type is the wrong half and is
// dropped. A declared type this check has no rule for — a name outside the
// JSON-Schema seven — is not in conflict: there is nothing to compare
// against, and the type rides as written. A type written as a list cannot be
// checked either, and is treated as the conflict it reads as.
func enumConflictsWithType(schema map[string]any) bool {
	values, ok := jsonx.AsSlice(schema["enum"])
	if !ok {
		return false
	}
	declared, present := schema["type"]
	if !present || declared == nil {
		return false
	}
	name, isString := declared.(string)
	if !isString {
		return true
	}
	switch name {
	case "null", "boolean", "integer", "number", "string", "array", "object":
	default:
		return false
	}
	for _, v := range values {
		if !matchesJSONType(name, v) {
			return true
		}
	}
	return false
}

func matchesJSONType(declared string, v any) bool {
	switch declared {
	case "null":
		return v == nil
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "integer":
		switch n := v.(type) {
		// A JSON number is an integer exactly when it was written as one:
		// the decoded literal carries its own shape, the way the reference's
		// reader tells int from float apart.
		case json.Number:
			return !strings.ContainsAny(n.String(), ".eE")
		case float64:
			return n == float64(int64(n))
		case int, int64:
			return true
		}
		return false
	case "number":
		switch v.(type) {
		case json.Number, float64, int, int64:
			return true
		}
		return false
	case "string":
		_, ok := v.(string)
		return ok
	case "array":
		_, ok := jsonx.AsSlice(v)
		return ok
	case "object":
		_, ok := jsonx.AsMap(v)
		return ok
	}
	return false
}

func deepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = deepCopy(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = deepCopy(val)
		}
		return out
	default:
		return v
	}
}

// inlineOutputSchemaDefs resolves the definitions a structured-output schema
// carries into the schema itself: the Messages target compiles the schema as
// written and refuses a $ref it cannot follow, so every reference naming a
// definition is replaced by that definition — a circular or unknown reference
// is left for the target to judge — and the defs blocks are removed. The
// expansion counts against the same byte budget an input schema does.
func inlineOutputSchemaDefs(schema map[string]any) error {
	var defs map[string]any
	if d, ok := jsonx.AsMap(schema["definitions"]); ok {
		defs = d
	}
	delete(schema, "definitions")
	if d, ok := jsonx.AsMap(schema["$defs"]); ok {
		defs = d
	}
	delete(schema, "$defs")
	if len(defs) == 0 {
		return nil
	}
	return unpackSchemaRefs(schema, defs)
}

// strictifySchemaForChat rewrites an output schema to what the chat dialect's
// strict mode demands: every object schema states additionalProperties:false
// and names all of its properties as required, and the combiners and array
// items are walked so nested objects comply too.
func strictifySchemaForChat(schema map[string]any) {
	if jsonx.GetString(schema, "type") == "object" {
		if props, ok := jsonx.AsMap(schema["properties"]); ok {
			schema["additionalProperties"] = false
			required := make([]any, 0, len(props))
			for k, v := range props {
				required = append(required, k)
				if pm, ok := jsonx.AsMap(v); ok {
					strictifySchemaForChat(pm)
				}
			}
			schema["required"] = required
		}
	}
	if items, ok := jsonx.AsMap(schema["items"]); ok {
		strictifySchemaForChat(items)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if list, ok := jsonx.AsSlice(schema[key]); ok {
			for _, sub := range list {
				if sm, ok := jsonx.AsMap(sub); ok {
					strictifySchemaForChat(sm)
				}
			}
		}
	}
	for _, key := range []string{"$defs", "definitions"} {
		if defs, ok := jsonx.AsMap(schema[key]); ok {
			for _, v := range defs {
				if dm, ok := jsonx.AsMap(v); ok {
					strictifySchemaForChat(dm)
				}
			}
		}
	}
}
