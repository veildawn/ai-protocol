package geminichat

import (
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// This file narrows a caller's JSON Schema into the dialect the target wire
// accepts for a tool declaration or a structured-output response. The reference
// rewrites the schema through a fixed pipeline before the request goes out, and
// a strict upstream rejects the shape it did not produce, so the fold performs
// the same steps in the same order:
//
//  1. `$defs` is popped from the root and every `$ref` naming a definition is
//     inlined (a dangling pointer survives none of the later steps),
//  2. an `anyOf` member typed `null` is removed and the remaining members are
//     marked `nullable`,
//  3. a `type` written as an array becomes `anyOf`,
//  4. empty strings in an `enum` become null,
//  5. an `enum` on a non-string type is dropped,
//  6. `type: "array"` without `items` gets an empty object schema,
//  7. a schema with no type gets `type: "object"` and empty `properties` go,
//  8. every key outside the wire's schema vocabulary is dropped.
//
// `strict` and a false `additionalProperties` are removed even earlier, over
// the whole tools list, because the target refuses both spellings outright.

// targetSchemaKeys is the whole vocabulary the target wire's schema type
// accepts. Everything else is filtered out, which is why step 1 has to inline
// `$defs` first: the definition a `$ref` points at would otherwise be removed
// and leave a pointer to nothing.
var targetSchemaKeys = map[string]struct{}{
	"type": {}, "format": {}, "title": {}, "description": {}, "nullable": {},
	"default": {}, "items": {}, "minItems": {}, "maxItems": {}, "enum": {},
	"properties": {}, "propertyOrdering": {}, "required": {}, "minProperties": {},
	"maxProperties": {}, "minimum": {}, "maximum": {}, "minLength": {},
	"maxLength": {}, "pattern": {}, "example": {}, "anyOf": {},
}

// schemaMaxDepth caps recursion over a caller-supplied schema. The reference
// refuses one nested deeper than this rather than risk unbounded recursion, so
// the fold reports the same failure instead of passing a truncated schema on.
const schemaMaxDepth = 100

// schemaBuilder carries the first schema-building step that fails, so the
// recursion stays readable while the caller still gets one error.
type schemaBuilder struct {
	err error
}

// buildTargetSchema narrows one tool-declaration schema. parameters is written
// through — the caller hands it a copy it owns.
func buildTargetSchema(parameters map[string]any) (map[string]any, error) {
	defs, _ := jsonx.AsMap(parameters["$defs"])
	delete(parameters, "$defs")
	b := &schemaBuilder{}
	b.unpackDefs(parameters, defs, 0)
	b.anyOfNullToNullable(parameters, 0)
	b.convertSchemaTypes(parameters, 0)
	b.fixEnumEmptyStrings(parameters, 0)
	b.fixEnumTypes(parameters, 0)
	b.processItems(parameters, 0)
	b.addObjectType(parameters, 0)
	if b.err != nil {
		return nil, b.err
	}
	return filterSchemaFields(parameters, 0), nil
}

func (b *schemaBuilder) overflow(depth int, step string) bool {
	if b.err != nil {
		return true
	}
	if depth > schemaMaxDepth {
		b.err = fmt.Errorf("schema exceeded the %d-deep recursion cap while %s", schemaMaxDepth, step)
		return true
	}
	return false
}

// unpackDefs inlines `$ref`s whose last path segment names a definition in
// scope. The scope starts from the caller's defs (the root's own `$defs`,
// popped by the caller) and widens with the `$defs` and `definitions` each node
// and each inlined target carries. A reference already being resolved on the
// current path is circular and stays as written; an unknown name stays too —
// the target is the one to judge a pointer it cannot resolve.
func (b *schemaBuilder) unpackDefs(schema map[string]any, defs map[string]any, depth int) {
	if b.overflow(depth, "inlining $defs") {
		return
	}
	rootDefs := map[string]any{}
	for k, v := range defs {
		rootDefs[k] = v
	}
	type slot struct {
		node   any
		parent any
		key    any
		defs   map[string]any
		chain  map[string]struct{}
	}
	widen := func(base map[string]any, m map[string]any) map[string]any {
		out := make(map[string]any, len(base)+2)
		for k, v := range base {
			out[k] = v
		}
		for _, name := range []string{"$defs", "definitions"} {
			if d, ok := jsonx.AsMap(m[name]); ok {
				for k, v := range d {
					out[k] = v
				}
			}
		}
		return out
	}
	queue := []slot{{node: schema, defs: widen(rootDefs, schema), chain: map[string]struct{}{}}}
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
				tm, _ := jsonx.AsMap(target)
				resolved := deepCopy(target)
				replaceSlot(cur.parent, cur.key, resolved, schema)
				chain := make(map[string]struct{}, len(cur.chain)+1)
				for k := range cur.chain {
					chain[k] = struct{}{}
				}
				chain[name] = struct{}{}
				queue = append(queue, slot{node: resolved, parent: cur.parent, key: cur.key, defs: widen(cur.defs, tm), chain: chain})
				continue
			}
			currentDefs := cur.defs
			if d, ok := jsonx.AsMap(node["$defs"]); ok && len(d) > 0 {
				currentDefs = widen(cur.defs, node)
			} else if d, ok := jsonx.AsMap(node["definitions"]); ok && len(d) > 0 {
				currentDefs = widen(cur.defs, node)
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
}

// replaceSlot writes an inlined definition back through the container that held
// the reference. A nil parent means the schema itself was the reference.
func replaceSlot(parent, key, resolved any, root map[string]any) {
	switch p := parent.(type) {
	case map[string]any:
		if k, ok := key.(string); ok {
			p[k] = resolved
		}
	case []any:
		if i, ok := key.(int); ok && i >= 0 && i < len(p) {
			p[i] = resolved
		}
	case nil:
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
	if i := strings.LastIndex(ref, "/"); i >= 0 {
		return ref[i+1:]
	}
	return ref
}

// anyOfNullToNullable removes `null` members from an `anyOf` and marks the
// survivors `nullable`, the spelling the target accepts for an optional value.
// An `anyOf` that was nothing but null types has nothing left to send, and the
// reference refuses the schema outright rather than inventing a type.
func (b *schemaBuilder) anyOfNullToNullable(schema map[string]any, depth int) {
	if b.overflow(depth, "folding anyOf nulls") {
		return
	}
	if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok && schema["anyOf"] != nil {
		kept := make([]any, 0, len(anyof))
		containsNull := false
		for _, item := range anyof {
			m, isMap := jsonx.AsMap(item)
			if isMap {
				if t, _ := m["type"].(string); t == "null" {
					containsNull = true
					continue
				}
				if _, hasType := m["type"]; !hasType && len(m) == 0 {
					// An empty member names no schema at all; the object it
					// must satisfy is the only reading left.
					m["type"] = "object"
				}
			}
			kept = append(kept, item)
		}
		if len(kept) == 0 {
			b.err = fmt.Errorf("invalid schema: anyOf with only a null type")
			return
		}
		if containsNull {
			for _, item := range kept {
				if m, ok := jsonx.AsMap(item); ok {
					m["nullable"] = true
				}
			}
		}
		schema["anyOf"] = kept
	}
	if props, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, v := range props {
			if m, ok := jsonx.AsMap(v); ok {
				b.anyOfNullToNullable(m, depth+1)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		b.anyOfNullToNullable(m, depth+1)
	}
}

// convertSchemaTypes rewrites a `type` written as an array into `anyOf`, moving
// the object- and array-specific keywords into the branch they describe. A
// single-element array collapses back to the scalar spelling. Note this runs
// after anyOfNullToNullable, so the null member a type array produces is kept
// verbatim — the two steps are not idempotent with each other, and the
// reference relies on that order.
func (b *schemaBuilder) convertSchemaTypes(schema map[string]any, depth int) {
	if b.overflow(depth, "converting type arrays") {
		return
	}
	if raw, has := schema["type"]; has {
		if list, isList := jsonx.AsSlice(raw); isList {
			switch {
			case len(list) > 1:
				var anyOf []any
				for _, t := range list {
					s, ok := t.(string)
					if !ok {
						continue
					}
					switch s {
					case "null":
						anyOf = append(anyOf, map[string]any{"type": "null"})
					case "object", "array":
						item := map[string]any{"type": s}
						for _, field := range typeSpecificSchemaFields {
							if v, ok := schema[field]; ok {
								item[field] = deepCopy(v)
							}
						}
						anyOf = append(anyOf, item)
					default:
						anyOf = append(anyOf, map[string]any{"type": s})
					}
				}
				hasObjectOrArray := false
				for _, t := range list {
					if s, ok := t.(string); ok && (s == "object" || s == "array") {
						hasObjectOrArray = true
					}
				}
				if hasObjectOrArray {
					for _, field := range typeSpecificSchemaFields {
						delete(schema, field)
					}
				}
				schema["anyOf"] = anyOf
				delete(schema, "type")
			case len(list) == 1:
				schema["type"] = list[0]
			}
		} else if _, isStr := raw.(string); isStr {
			schema["type"] = raw
		}
	}
	if props, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, v := range props {
			if m, ok := jsonx.AsMap(v); ok {
				b.convertSchemaTypes(m, depth+1)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		b.convertSchemaTypes(m, depth+1)
	}
	if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok {
		for _, v := range anyof {
			if m, ok := jsonx.AsMap(v); ok {
				b.convertSchemaTypes(m, depth+1)
			}
		}
	}
}

// typeSpecificSchemaFields are the keywords that describe one kind of value, so
// when a type array fans out into anyOf branches each keyword follows its kind.
var typeSpecificSchemaFields = []string{
	"properties", "required", "additionalProperties", "items",
	"minItems", "maxItems", "minProperties", "maxProperties",
}

// fixEnumEmptyStrings turns an empty string inside an `enum` into null — the
// target refuses an empty enum member, and the null spelling keeps the slot.
func (b *schemaBuilder) fixEnumEmptyStrings(schema map[string]any, depth int) {
	if b.overflow(depth, "fixing empty enum strings") {
		return
	}
	if enum, ok := jsonx.AsSlice(schema["enum"]); ok {
		out := make([]any, 0, len(enum))
		for _, v := range enum {
			if s, isStr := v.(string); isStr && s == "" {
				out = append(out, nil)
				continue
			}
			out = append(out, v)
		}
		schema["enum"] = out
	}
	if props, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, v := range props {
			if m, ok := jsonx.AsMap(v); ok {
				b.fixEnumEmptyStrings(m, depth+1)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		b.fixEnumEmptyStrings(m, depth+1)
	}
}

// fixEnumTypes drops an `enum` from a schema whose type is not string: the
// target only accepts enums on string-typed fields, and an enum a different
// type cannot honour is a constraint the target would reject wholesale.
func (b *schemaBuilder) fixEnumTypes(schema map[string]any, depth int) {
	if b.overflow(depth, "fixing enum types") {
		return
	}
	if enum, ok := jsonx.AsSlice(schema["enum"]); ok && enum != nil {
		keep := false
		if t, ok := schema["type"].(string); ok && strings.EqualFold(t, "string") {
			keep = true
		} else if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok {
			for _, item := range anyof {
				m, isMap := jsonx.AsMap(item)
				if !isMap {
					continue
				}
				if t, ok := m["type"].(string); ok && strings.EqualFold(t, "string") {
					keep = true
					break
				}
			}
		}
		if !keep {
			delete(schema, "enum")
		}
	}
	if props, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, v := range props {
			if m, ok := jsonx.AsMap(v); ok {
				b.fixEnumTypes(m, depth+1)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		b.fixEnumTypes(m, depth+1)
	}
	if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok {
		for _, v := range anyof {
			if m, ok := jsonx.AsMap(v); ok {
				b.fixEnumTypes(m, depth+1)
			}
		}
	}
}

// processItems gives every `type: "array"` an `items` schema — the target
// requires one, and an empty or missing items object becomes the empty object
// schema rather than a rejection.
func (b *schemaBuilder) processItems(schema map[string]any, depth int) {
	if b.overflow(depth, "completing array items") {
		return
	}
	if t, ok := schema["type"].(string); ok && strings.EqualFold(t, "array") {
		missing := true
		if raw, has := schema["items"]; has {
			if m, isMap := jsonx.AsMap(raw); !isMap || len(m) == 0 {
				missing = true
			} else {
				missing = false
			}
		}
		if missing {
			schema["items"] = map[string]any{"type": "object"}
		}
	}
	for _, v := range schema {
		if m, ok := jsonx.AsMap(v); ok {
			b.processItems(m, depth+1)
		} else if list, ok := jsonx.AsSlice(v); ok {
			for _, item := range list {
				if m, ok := jsonx.AsMap(item); ok {
					b.processItems(m, depth+1)
				}
			}
		}
	}
}

// addObjectType states `type: "object"` wherever the schema implies one — every
// function parameter must be an object, and a node with properties but no type
// would otherwise be rejected. Empty `properties` are removed rather than sent:
// the target refuses an object with no members at all.
func (b *schemaBuilder) addObjectType(schema map[string]any, depth int) {
	if b.overflow(depth, "adding object types") {
		return
	}
	_, hasType := schema["type"]
	_, hasAnyOf := schema["anyOf"]
	_, hasOneOf := schema["oneOf"]
	_, hasAllOf := schema["allOf"]
	if !hasType && !hasAnyOf && !hasOneOf && !hasAllOf {
		schema["type"] = "object"
	}
	if props, ok := jsonx.AsMap(schema["properties"]); ok && schema["properties"] != nil {
		if v, has := schema["required"]; has && v == nil {
			delete(schema, "required")
		}
		if len(props) == 0 {
			delete(schema, "properties")
			delete(schema, "required")
			schema["type"] = "object"
		} else {
			schema["type"] = "object"
			for _, v := range props {
				if m, ok := jsonx.AsMap(v); ok {
					b.addObjectType(m, depth+1)
				}
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		b.addObjectType(m, depth+1)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if list, ok := jsonx.AsSlice(schema[key]); ok {
			for _, v := range list {
				if m, ok := jsonx.AsMap(v); ok {
					b.addObjectType(m, depth+1)
				}
			}
		}
	}
}

// filterSchemaFields drops every key outside the target's schema vocabulary.
// A node carrying `anyOf` is narrowed to just that member — the target refuses
// the siblings — with the node's own title and description copied into each
// branch so the words are not lost.
func filterSchemaFields(schema map[string]any, depth int) map[string]any {
	if depth > schemaMaxDepth {
		return schema
	}
	if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok && len(anyof) > 0 {
		title, hasTitle := schema["title"]
		desc, hasDesc := schema["description"]
		if hasTitle || hasDesc {
			allDicts := true
			for _, item := range anyof {
				if _, isMap := jsonx.AsMap(item); !isMap {
					allDicts = false
					break
				}
			}
			if allDicts {
				for _, item := range anyof {
					m, _ := jsonx.AsMap(item)
					if hasTitle {
						m["title"] = title
					}
					if hasDesc {
						m["description"] = desc
					}
				}
			}
		}
		schema = map[string]any{"anyOf": anyof}
	}
	result := map[string]any{}
	for k, v := range schema {
		if _, ok := targetSchemaKeys[k]; !ok {
			continue
		}
		switch k {
		case "properties":
			if props, ok := jsonx.AsMap(v); ok {
				out := map[string]any{}
				for pk, pv := range props {
					if pm, ok := jsonx.AsMap(pv); ok {
						out[pk] = filterSchemaFields(pm, depth+1)
						continue
					}
					out[pk] = pv
				}
				result[k] = out
				continue
			}
		case "format":
			if s, ok := v.(string); ok && (s == "enum" || s == "date-time") {
				result[k] = v
			}
			continue
		case "items":
			if m, ok := jsonx.AsMap(v); ok {
				result[k] = filterSchemaFields(m, depth+1)
				continue
			}
		case "anyOf":
			if list, ok := jsonx.AsSlice(v); ok {
				out := make([]any, 0, len(list))
				for _, item := range list {
					if m, ok := jsonx.AsMap(item); ok {
						out = append(out, filterSchemaFields(m, depth+1))
						continue
					}
					out = append(out, item)
				}
				result[k] = out
				continue
			}
		}
		result[k] = v
	}
	return result
}

// removeStrict deletes every `strict` key, anywhere in the value. The target
// has no strict spelling — the reference strips it from tool schemas and from
// a structured-output schema before either is sent.
func removeStrict(v any) {
	switch t := v.(type) {
	case map[string]any:
		delete(t, "strict")
		for _, v := range t {
			removeStrict(v)
		}
	case []any:
		for _, item := range t {
			removeStrict(item)
		}
	}
}

// removeAdditionalProperties deletes `additionalProperties` where it is
// literally false. The target rejects the keyword in that spelling; a true or
// descriptive value is left for the key whitelist to judge.
func removeAdditionalProperties(v any) {
	switch t := v.(type) {
	case map[string]any:
		if b, ok := t["additionalProperties"].(bool); ok && !b {
			delete(t, "additionalProperties")
		}
		for _, v := range t {
			removeAdditionalProperties(v)
		}
	case []any:
		for _, item := range t {
			removeAdditionalProperties(item)
		}
	}
}

// deepCopy duplicates a decoded-JSON value so a schema rewrite never writes
// through the caller's map — the library's folds are pure functions by
// contract, and the reference's own builder mutates in place.
func deepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = deepCopy(val)
		}
		return out
	case []any:
		out := make([]any, 0, len(t))
		for _, val := range t {
			out = append(out, deepCopy(val))
		}
		return out
	default:
		return v
	}
}
