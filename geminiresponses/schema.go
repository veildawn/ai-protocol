package geminiresponses

import (
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// targetSchemaKeys is the whole vocabulary the target wire's schema object
// accepts. Every other key is dropped from a tool's parameters, which is why
// JSON-Schema-only keys like $schema, additionalProperties or strict never
// reach the upstream.
var targetSchemaKeys = map[string]bool{
	"type": true, "format": true, "title": true, "description": true,
	"nullable": true, "default": true, "items": true, "minItems": true,
	"maxItems": true, "enum": true, "properties": true, "propertyOrdering": true,
	"required": true, "minProperties": true, "maxProperties": true,
	"minimum": true, "maximum": true, "minLength": true, "maxLength": true,
	"pattern": true, "example": true, "anyOf": true,
}

// toolParameters readies a Responses function tool's parameters for the target
// wire, the way the reference's request fold does: additionalProperties:false
// and strict are stripped everywhere, then the schema is normalised into the
// shape the target accepts ($defs inlined, a null alternative folded into
// nullable, type arrays widened to anyOf, enum values fixed, array items
// completed, a missing type spelled out as object, and every key outside the
// accepted vocabulary removed). The schema is copied first — the caller's map
// is never written through, which is this library's pure-function contract.
func toolParameters(params any) (map[string]any, error) {
	var m map[string]any
	if src, ok := params.(map[string]any); ok {
		m = deepCopyValue(src).(map[string]any)
	} else {
		// An absent, empty or non-object parameters reads as the empty object;
		// the reference's bridge also replaces a blank string with one.
		m = map[string]any{}
	}
	if len(m) == 0 {
		m["type"] = "object"
	} else if _, ok := m["type"]; !ok {
		m["type"] = "object"
	}
	removeFalseAdditionalProperties(m)
	removeStrictKeys(m)
	defs := map[string]any{}
	if d, ok := m["$defs"].(map[string]any); ok {
		defs = d
		delete(m, "$defs")
	}
	unpackSchemaRefs(m, defs)
	if err := convertAnyOfNullToNullable(m); err != nil {
		return nil, err
	}
	convertSchemaTypes(m)
	fixEnumEmptyStrings(m)
	fixEnumTypes(m)
	processSchemaItems(m)
	addSchemaObjectType(m)
	return filterSchemaFields(m), nil
}

// deepCopyValue duplicates a decoded JSON value so later passes can rewrite
// the copy without the caller observing the change.
func deepCopyValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, vv := range t {
			out[k] = deepCopyValue(vv)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, vv := range t {
			out[i] = deepCopyValue(vv)
		}
		return out
	default:
		return v
	}
}

// removeStrictKeys deletes every `strict` key at any depth. The target wire
// rejects the key outright, and the reference strips it from both the tool
// envelope and the schema body before building declarations — including a
// schema *property* named strict, which is why the walk does not special-case
// `properties`.
func removeStrictKeys(v any) {
	switch t := v.(type) {
	case map[string]any:
		delete(t, "strict")
		for _, vv := range t {
			removeStrictKeys(vv)
		}
	case []any:
		for _, vv := range t {
			removeStrictKeys(vv)
		}
	}
}

// removeFalseAdditionalProperties deletes additionalProperties only where it
// is exactly false — the spelling the target wire rejects. A true value or a
// schema object rides along until the key whitelist drops it.
func removeFalseAdditionalProperties(v any) {
	switch t := v.(type) {
	case map[string]any:
		if b, ok := t["additionalProperties"].(bool); ok && !b {
			delete(t, "additionalProperties")
		}
		for _, vv := range t {
			removeFalseAdditionalProperties(vv)
		}
	case []any:
		for _, vv := range t {
			removeFalseAdditionalProperties(vv)
		}
	}
}

// unpackSchemaRefs replaces each {"$ref": "#/$defs/X"} whose last path segment
// names a definition in scope with a copy of that definition, walking the
// whole tree so references hidden under items/anyOf/oneOf resolve too. The
// scope widens with the $defs/definitions each node carries; a reference
// already being resolved on the current chain is circular and stays as
// written, and an unknown reference is left untouched.
func unpackSchemaRefs(root map[string]any, defs map[string]any) {
	rootDefs := map[string]any{}
	for k, v := range defs {
		rootDefs[k] = v
	}
	for _, key := range []string{"$defs", "definitions"} {
		if d, ok := root[key].(map[string]any); ok {
			for k, v := range d {
				rootDefs[k] = v
			}
		}
	}
	type slot struct {
		node  any
		set   func(any) // writes the replacement back into the parent (root for the schema itself)
		defs  map[string]any
		chain map[string]bool
	}
	// The root is rewritten in place (cleared and refilled) so the caller's
	// map keeps the resolved body — the same move the reference makes when the
	// schema itself is a reference.
	setRoot := func(v any) {
		if rm, ok := v.(map[string]any); ok {
			for k := range root {
				delete(root, k)
			}
			for k, vv := range rm {
				root[k] = vv
			}
		}
	}
	queue := []slot{{node: root, set: setRoot, defs: rootDefs, chain: map[string]bool{}}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		switch node := cur.node.(type) {
		case map[string]any:
			if ref := jsonx.GetString(node, "$ref"); ref != "" {
				name := lastPathSegment(ref)
				if cur.chain[name] {
					continue
				}
				target, known := cur.defs[name]
				if !known || target == nil {
					continue
				}
				tm, ok := target.(map[string]any)
				if !ok {
					continue
				}
				childDefs := map[string]any{}
				for k, v := range cur.defs {
					childDefs[k] = v
				}
				for _, key := range []string{"$defs", "definitions"} {
					if d, ok := tm[key].(map[string]any); ok {
						for k, v := range d {
							childDefs[k] = v
						}
					}
				}
				resolved := deepCopyValue(tm)
				cur.set(resolved)
				chain := map[string]bool{}
				for k := range cur.chain {
					chain[k] = true
				}
				chain[name] = true
				queue = append(queue, slot{node: resolved, set: cur.set, defs: childDefs, chain: chain})
				continue
			}
			currentDefs := map[string]any{}
			for k, v := range cur.defs {
				currentDefs[k] = v
			}
			for _, key := range []string{"$defs", "definitions"} {
				if d, ok := node[key].(map[string]any); ok {
					for k, v := range d {
						currentDefs[k] = v
					}
				}
			}
			for k, vv := range node {
				key, val := k, vv
				queue = append(queue, slot{
					node:  val,
					set:   func(v any) { node[key] = v },
					defs:  currentDefs,
					chain: cur.chain,
				})
			}
		case []any:
			for idx, vv := range node {
				i, val := idx, vv
				queue = append(queue, slot{
					node:  val,
					set:   func(v any) { node[i] = v },
					defs:  cur.defs,
					chain: cur.chain,
				})
			}
		}
	}
}

// lastPathSegment takes the definition name off a JSON pointer's tail.
func lastPathSegment(ref string) string {
	if i := strings.LastIndex(ref, "/"); i >= 0 {
		return ref[i+1:]
	}
	return ref
}

// convertAnyOfNullToNullable folds a null alternative into `nullable` on the
// remaining branches, the form the target wire wants for optional values. An
// anyOf holding nothing but null has nothing left to send, which the reference
// refuses rather than emit an invalid schema.
func convertAnyOfNullToNullable(schema map[string]any) error {
	anyOf, ok := schema["anyOf"].([]any)
	if !ok {
		return descendSchema(schema, convertAnyOfNullToNullable)
	}
	containsNull := false
	kept := make([]any, 0, len(anyOf))
	for _, it := range anyOf {
		if m, ok := it.(map[string]any); ok {
			if jsonx.GetString(m, "type") == "null" {
				containsNull = true
				continue
			}
			if _, has := m["type"]; !has && len(m) == 0 {
				m["type"] = "object"
			}
		}
		kept = append(kept, it)
	}
	if len(kept) == 0 {
		return fmt.Errorf("anyOf schema with only a null type has no non-null branch to send")
	}
	if containsNull {
		for _, it := range kept {
			if m, ok := it.(map[string]any); ok {
				m["nullable"] = true
			}
		}
	}
	schema["anyOf"] = kept
	return descendSchema(schema, convertAnyOfNullToNullable)
}

// convertSchemaTypes widens a type array into anyOf — the target takes one
// type per schema node — moving the object/array-only keywords into the
// branch that needs them. A single-element array collapses to the scalar.
func convertSchemaTypes(schema map[string]any) {
	if tv, ok := schema["type"]; ok {
		switch t := tv.(type) {
		case []any:
			if len(t) > 1 {
				typeSpecific := []string{"properties", "required", "additionalProperties",
					"items", "minItems", "maxItems", "minProperties", "maxProperties"}
				anyOf := make([]any, 0, len(t))
				hasObjOrArr := false
				for _, e := range t {
					s, ok := e.(string)
					if !ok {
						continue
					}
					switch s {
					case "null":
						anyOf = append(anyOf, map[string]any{"type": "null"})
					case "object", "array":
						hasObjOrArr = true
						item := map[string]any{"type": s}
						for _, f := range typeSpecific {
							if v, ok := schema[f]; ok {
								item[f] = deepCopyValue(v)
							}
						}
						anyOf = append(anyOf, item)
					default:
						anyOf = append(anyOf, map[string]any{"type": s})
					}
				}
				if hasObjOrArr {
					for _, f := range typeSpecific {
						delete(schema, f)
					}
				}
				schema["anyOf"] = anyOf
				delete(schema, "type")
			} else if len(t) == 1 {
				schema["type"] = t[0]
			}
		}
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, v := range props {
			if m, ok := v.(map[string]any); ok {
				convertSchemaTypes(m)
			}
		}
	}
	if m, ok := schema["items"].(map[string]any); ok {
		convertSchemaTypes(m)
	}
	if anyOf, ok := schema["anyOf"].([]any); ok {
		for _, v := range anyOf {
			if m, ok := v.(map[string]any); ok {
				convertSchemaTypes(m)
			}
		}
	}
}

// fixEnumEmptyStrings turns an empty string inside an enum into an explicit
// null: the target rejects empty enum members but accepts null as "no value".
func fixEnumEmptyStrings(schema map[string]any) {
	if enum, ok := schema["enum"].([]any); ok {
		for i, v := range enum {
			if s, ok := v.(string); ok && s == "" {
				enum[i] = nil
			}
		}
	}
	descendSchemaValues(schema, fixEnumEmptyStrings)
}

// fixEnumTypes drops an enum the target would reject: enums are only legal on
// string-typed fields, so a non-string type (and no string branch in anyOf)
// loses the constraint instead of failing the whole request.
func fixEnumTypes(schema map[string]any) {
	if _, ok := schema["enum"].([]any); ok {
		keep := false
		if ts, ok := schema["type"].(string); ok && strings.EqualFold(ts, "string") {
			keep = true
		}
		if !keep {
			if anyOf, ok := schema["anyOf"].([]any); ok {
				for _, it := range anyOf {
					if m, ok := it.(map[string]any); ok {
						if ts, ok := m["type"].(string); ok && strings.EqualFold(ts, "string") {
							keep = true
							break
						}
					}
				}
			}
		}
		if !keep {
			delete(schema, "enum")
		}
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, v := range props {
			if m, ok := v.(map[string]any); ok {
				fixEnumTypes(m)
			}
		}
	}
	if m, ok := schema["items"].(map[string]any); ok {
		fixEnumTypes(m)
	}
	if anyOf, ok := schema["anyOf"].([]any); ok {
		for _, v := range anyOf {
			if m, ok := v.(map[string]any); ok {
				fixEnumTypes(m)
			}
		}
	}
}

// processSchemaItems completes an array schema: the target requires `items`
// whenever type is array, so a missing or empty one becomes a generic object.
func processSchemaItems(schema map[string]any) {
	if ts, ok := schema["type"].(string); ok && strings.EqualFold(ts, "array") {
		items, present := schema["items"]
		empty := !present
		if present {
			if m, ok := items.(map[string]any); ok && len(m) == 0 {
				empty = true
			}
		}
		if empty {
			schema["items"] = map[string]any{"type": "object"}
		}
	}
	for _, v := range schema {
		if m, ok := v.(map[string]any); ok {
			processSchemaItems(m)
		} else if l, ok := v.([]any); ok {
			for _, it := range l {
				if m, ok := it.(map[string]any); ok {
					processSchemaItems(m)
				}
			}
		}
	}
}

// addSchemaObjectType states `type: object` wherever no type alternative is
// spelled out, drops a null `required`, and removes an empty `properties`
// while keeping the object type — the target rejects an object with no
// properties block but accepts a bare object.
func addSchemaObjectType(schema map[string]any) {
	if _, hasType := schema["type"]; !hasType {
		_, hasAnyOf := schema["anyOf"]
		_, hasOneOf := schema["oneOf"]
		_, hasAllOf := schema["allOf"]
		if !hasAnyOf && !hasOneOf && !hasAllOf {
			schema["type"] = "object"
		}
	}
	if props, ok := schema["properties"].(map[string]any); ok {
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
				if m, ok := v.(map[string]any); ok {
					addSchemaObjectType(m)
				}
			}
		}
	}
	if m, ok := schema["items"].(map[string]any); ok {
		addSchemaObjectType(m)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if vs, ok := schema[key].([]any); ok {
			for _, v := range vs {
				if m, ok := v.(map[string]any); ok {
					addSchemaObjectType(m)
				}
			}
		}
	}
}

// filterSchemaFields keeps only the keys the target's schema vocabulary
// accepts. A node carrying anyOf keeps nothing else — the target rejects the
// combination — but its title and description are pushed down into each
// branch so the wording survives.
func filterSchemaFields(schema map[string]any) map[string]any {
	return filterSchemaNode(filterAnyOfFields(schema))
}

func filterSchemaNode(schema map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range schema {
		if !targetSchemaKeys[k] {
			continue
		}
		switch k {
		case "properties":
			if pm, ok := v.(map[string]any); ok {
				filtered := map[string]any{}
				for pk, pv := range pm {
					if m, ok := pv.(map[string]any); ok {
						filtered[pk] = filterSchemaNode(m)
					} else {
						filtered[pk] = pv
					}
				}
				out[k] = filtered
			} else {
				out[k] = v
			}
		case "format":
			if s, ok := v.(string); ok && (s == "enum" || s == "date-time") {
				out[k] = v
			}
		case "items":
			if m, ok := v.(map[string]any); ok {
				out[k] = filterSchemaNode(m)
			} else {
				out[k] = v
			}
		case "anyOf":
			if l, ok := v.([]any); ok {
				filtered := make([]any, 0, len(l))
				for _, it := range l {
					if m, ok := it.(map[string]any); ok {
						filtered = append(filtered, filterSchemaNode(m))
					} else {
						filtered = append(filtered, it)
					}
				}
				out[k] = filtered
			} else {
				out[k] = v
			}
		default:
			out[k] = v
		}
	}
	return out
}

// filterAnyOfFields reduces a node that carries anyOf to that anyOf alone,
// pushing title and description into each branch on the way.
func filterAnyOfFields(schema map[string]any) map[string]any {
	anyOf, ok := schema["anyOf"].([]any)
	if !ok || len(anyOf) == 0 {
		return schema
	}
	title, hasTitle := schema["title"]
	desc, hasDesc := schema["description"]
	allMaps := true
	for _, it := range anyOf {
		if _, ok := it.(map[string]any); !ok {
			allMaps = false
			break
		}
	}
	if allMaps {
		for _, it := range anyOf {
			m := it.(map[string]any)
			if hasTitle && title != nil && title != "" {
				m["title"] = title
			}
			if hasDesc && desc != nil && desc != "" {
				m["description"] = desc
			}
		}
	}
	return map[string]any{"anyOf": anyOf}
}

// descendSchema applies fn to each property schema and the items schema — the
// recursion shape the reference's schema passes share.
func descendSchema(schema map[string]any, fn func(map[string]any) error) error {
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, v := range props {
			if m, ok := v.(map[string]any); ok {
				if err := fn(m); err != nil {
					return err
				}
			}
		}
	}
	if m, ok := schema["items"].(map[string]any); ok {
		if err := fn(m); err != nil {
			return err
		}
	}
	return nil
}

// descendSchemaValues applies fn to every map value of the node.
func descendSchemaValues(schema map[string]any, fn func(map[string]any)) {
	for _, v := range schema {
		if m, ok := v.(map[string]any); ok {
			fn(m)
		}
	}
}
