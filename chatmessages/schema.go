package chatmessages

import "github.com/veildawn/ai-protocol/internal/jsonx"

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
