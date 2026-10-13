package chatmessages

import (
	"fmt"
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
// accepted set removed.
func sanitizeInputSchemaForAnthropic(schema any) map[string]any {
	out, _ := jsonx.AsMap(schema)
	if out == nil {
		out = map[string]any{}
	} else {
		out = jsonx.CloneMap(out)
	}
	if jsonx.GetString(out, "type") != "object" {
		out["type"] = "object"
	}
	if _, ok := out["properties"]; !ok {
		out["properties"] = map[string]any{}
	}
	unpackLegacyDefs(out)
	for key := range out {
		if _, ok := anthropicSchemaKeys[key]; !ok {
			delete(out, key)
		}
	}
	return out
}

// unpackLegacyDefs inlines `$ref`s backed by draft-04 `definitions` or OpenAPI
// `components.schemas` into the schema itself, then drops those blocks. Only
// `$defs` survives as a ref target, so a legacy ref left in place would point
// at a definition the target dialect no longer carries.
func unpackLegacyDefs(schema map[string]any) {
	defs := map[string]any{}
	if components, ok := jsonx.AsMap(schema["components"]); ok {
		if schemas, ok := jsonx.AsMap(components["schemas"]); ok {
			for k, v := range schemas {
				defs[k] = v
			}
		}
	}
	if definitions, ok := jsonx.AsMap(schema["definitions"]); ok {
		for k, v := range definitions {
			defs[k] = v
		}
	}
	delete(schema, "components")
	delete(schema, "definitions")
	if len(defs) == 0 {
		return
	}
	// Each entry is (node, parent, key): a resolved node is written back through
	// the container that held the ref.
	type slot struct {
		node   any
		parent any
		key    any
	}
	queue := []slot{{node: schema, parent: nil, key: nil}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		switch node := cur.node.(type) {
		case map[string]any:
			if ref := jsonx.GetString(node, "$ref"); ref != "" {
				target, ok := jsonx.AsMap(defs[lastPathSegment(ref)])
				if !ok {
					continue
				}
				resolved := deepCopy(target)
				replaceSlot(cur.parent, cur.key, resolved, schema)
				queue = append(queue, slot{node: resolved, parent: cur.parent, key: cur.key})
				continue
			}
			for k, v := range node {
				queue = append(queue, slot{node: v, parent: node, key: k})
			}
		case []any:
			for i, v := range node {
				queue = append(queue, slot{node: v, parent: node, key: i})
			}
		}
	}
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
			if list, ok := jsonx.AsSlice(value); ok {
				var filtered []any
				for _, item := range list {
					filtered = append(filtered, filterOutputSchemaForAnthropic(item))
				}
				out["anyOf"] = append(existingList(out["anyOf"]), filtered...)
				continue
			}
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
// serializing a sub-schema so the note says what was actually required.
func formatConstraint(value any) string {
	switch value.(type) {
	case map[string]any, []any:
		if b, err := jsonx.Marshal(value); err == nil {
			return string(b)
		}
	}
	return jsonx.String(value)
}

// enumConflictsWithType reports whether a schema's enum cannot match its
// declared type, in which case the declared type is the wrong half and is
// dropped.
func enumConflictsWithType(schema map[string]any) bool {
	values, ok := jsonx.AsSlice(schema["enum"])
	if !ok {
		return false
	}
	declared := jsonx.GetString(schema, "type")
	if declared == "" {
		return false
	}
	for _, v := range values {
		if !matchesJSONType(declared, v) {
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
		case float64:
			return n == float64(int64(n))
		case int, int64:
			return true
		}
		return false
	case "number":
		switch v.(type) {
		case float64, int, int64:
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
