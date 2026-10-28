// Package schemanorm re-spells the gemini wire's uppercase schema enums into
// the lowercase JSON Schema spellings every OpenAI-family dialect requires.
package schemanorm

import "github.com/veildawn/ai-protocol/internal/jsonx"

// The gemini wire accepts SCREAMING_SNAKE type enums ("OBJECT", "STRING", …)
// on a functionDeclarations schema — its native spelling. Every dialect a
// gemini request folds onto (Chat, Messages, Responses) validates `type`
// against the lowercase JSON Schema vocabulary and refuses the uppercase
// spelling with a hard 400, so each out-of-gemini fold has to normalize the
// schema it relays. The rewrite is pure dialect mechanics — no host policy —
// which is why it lives in the codec rather than at a gateway seam.

// A declaration's schema rides one of three members. `parameters` is the
// wire's original OpenAPI-subset Schema; `parametersJsonSchema` and its snake
// spelling carry a standard JSON Schema instead — the member the official
// SDKs now document, and protobuf's JSON mapping takes lowerCamel and snake
// alike, so a fold that reads only `parameters` hands the target a schema-less
// declaration whenever a client states the newer member. SchemaOf resolves
// the three spellings so every out-of-gemini fold reads one shape; the legacy
// member wins when a declaration states both, because it is the one the
// reference bridge reads first.

// schemaMembers lists the declaration's schema members in precedence order.
var schemaMembers = []string{"parameters", "parameters_json_schema", "parametersJsonSchema"}

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

// SchemaOf returns the declaration's parameters schema, resolved across the
// wire's three spellings of the member. A stated-but-null schema counts as
// absent, so a fold that must state one anyway falls to its own default.
func SchemaOf(decl map[string]any) (any, bool) {
	for _, member := range schemaMembers {
		if v, ok := decl[member]; ok && v != nil {
			return v, true
		}
	}
	return nil, false
}

// NormalizeTree walks a decoded tool-declaration list re-spelling every
// "type" member under each declaration's parameters schema, whichever member
// carries it. It reports whether anything moved, so a caller that normalizes
// in place can keep the original bytes when nothing did.
func NormalizeTree(declarations []any) bool {
	changed := false
	for _, raw := range declarations {
		dm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if params, ok := SchemaOf(dm); ok {
			if node, isMap := params.(map[string]any); isMap {
				changed = normalizeSchemaNode(node) || changed
			}
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

// responseSchemaMembers lists the generation config's structured-output
// schema members in precedence order: responseJsonSchema carries the standard
// JSON Schema, responseSchema the wire's original member, and a config that
// states both means the newer one.
var responseSchemaMembers = []string{"response_json_schema", "response_schema"}

// ResponseSchemaOf reads the generation config's structured-output schema and
// returns it re-spelled: the same enum normalization NormalizeTree applies to
// a declaration's parameters, on the config's own schema, in place. The
// official SDKs state the type enums in the gemini wire's SCREAMING spelling
// there too, and every target dialect validates lowercase — a strict upstream
// answers the uppercase schema with a hard failure, the way it refuses an
// uppercase declaration, so the response schema crosses a fold normalized or
// not at all.
//
// A stated-but-empty schema counts as absent: the structured-output config
// then degrades to the target's plain json_object request, the same reading
// the folds apply to a schema member that is missing entirely.
func ResponseSchemaOf(gc map[string]any) (map[string]any, bool) {
	for _, member := range responseSchemaMembers {
		if sm, ok := jsonx.AsMap(gc[member]); ok && len(sm) > 0 {
			normalizeSchemaNode(sm)
			return sm, true
		}
	}
	return nil, false
}
