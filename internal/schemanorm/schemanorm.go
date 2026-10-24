// Package schemanorm re-spells the gemini wire's uppercase schema enums into
// the lowercase JSON Schema spellings every OpenAI-family dialect requires.
package schemanorm

// The gemini wire accepts SCREAMING_SNAKE type enums ("OBJECT", "STRING", …)
// on a functionDeclarations schema — its native spelling. Every dialect a
// gemini request folds onto (Chat, Messages, Responses) validates `type`
// against the lowercase JSON Schema vocabulary and refuses the uppercase
// spelling with a hard 400, so each out-of-gemini fold has to normalize the
// schema it relays. The rewrite is pure dialect mechanics — no host policy —
// which is why it lives in the codec rather than at a gateway seam.

// geminiEnumSpellings maps the gemini wire's type enums to the lowercase
// spellings. Unmapped keys pass through untouched.
var geminiEnumSpellings = map[string]string{
	"OBJECT":  "object",
	"ARRAY":   "array",
	"STRING":  "string",
	"NUMBER":  "number",
	"INTEGER": "integer",
	"BOOLEAN": "boolean",
}

// NormalizeTree walks a decoded tool-declaration list re-spelling every
// "type" member under each declaration's parameters schema. It reports
// whether anything moved, so a caller that normalizes in place can keep the
// original bytes when nothing did.
func NormalizeTree(declarations []any) bool {
	changed := false
	for _, raw := range declarations {
		dm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if params, ok := dm["parameters"].(map[string]any); ok {
			changed = normalizeSchemaNode(params) || changed
		}
	}
	return changed
}

// normalizeSchemaNode lowercases a "type" member and recurses into the
// schema's child nodes.
func normalizeSchemaNode(node map[string]any) bool {
	changed := false
	if t, ok := node["type"].(string); ok {
		if lower, mapped := geminiEnumSpellings[t]; mapped {
			node["type"] = lower
			changed = true
		}
	}
	switch items := node["items"].(type) {
	case map[string]any:
		changed = normalizeSchemaNode(items) || changed
	case []any:
		for _, raw := range items {
			if child, ok := raw.(map[string]any); ok {
				changed = normalizeSchemaNode(child) || changed
			}
		}
	}
	// properties maps a field name to its schema node.
	if props, ok := node["properties"].(map[string]any); ok {
		for _, raw := range props {
			if child, ok := raw.(map[string]any); ok {
				changed = normalizeSchemaNode(child) || changed
			}
		}
	}
	// anyOf / oneOf / allOf branches carry the same node shape.
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if branches, ok := node[key].([]any); ok {
			for _, raw := range branches {
				if child, ok := raw.(map[string]any); ok {
					changed = normalizeSchemaNode(child) || changed
				}
			}
		}
	}
	return changed
}
