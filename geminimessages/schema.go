package geminimessages

// This file carries the schema and wire helpers the two directions share: the
// schema builder the target wire accepts for tool parameters, the recursive
// strip the reference applies before building, the strict copy a structured
// output takes, and the JSON spelling the reference writes tool arguments in.

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// openaiMaxToolName is the tool-name cap the reference's Messages bridge
// enforces before the target fold reads the name: a longer name is rewritten to
// a 55-character prefix plus an 8-character digest of the whole name, so two
// long names that share a prefix still land on distinct wire names.
const (
	openaiMaxToolName  = 64
	toolNameHashLength = 8
	toolNamePrefix     = openaiMaxToolName - toolNameHashLength - 1
)

// truncateToolName rewrites a Messages tool name the way the reference's bridge
// does: names within the cap pass through, longer ones become
// prefix_digest.
func truncateToolName(name string) string {
	if len(name) <= openaiMaxToolName {
		return name
	}
	sum := sha256.Sum256([]byte(name))
	h := hex.EncodeToString(sum[:])[:toolNameHashLength]
	prefix := name
	if len(prefix) > toolNamePrefix {
		prefix = prefix[:toolNamePrefix]
	}
	return prefix + "_" + h
}

// thoughtSignatureSeparator marks the suffix the reference embeds a thought
// signature in a tool call id with; the Messages id pattern cannot carry it, so
// the fold moves it off the id and onto the block.
const thoughtSignatureSeparator = "__thought__"

// stripThoughtSignature splits a tool call id into its base and the
// thought-signature suffix, when one rides it.
func stripThoughtSignature(id string) (base, sig string) {
	if i := strings.Index(id, thoughtSignatureSeparator); i >= 0 {
		return id[:i], id[i+len(thoughtSignatureSeparator):]
	}
	return id, ""
}

// normalizeToolUseID makes a tool call id legal on the Messages wire, whose ids
// match ^[a-zA-Z0-9_-]+$: a thought-signature suffix is stripped first, every
// remaining character outside the class becomes '_', and an id with nothing
// left is named.
func normalizeToolUseID(id string) string {
	base, _ := stripThoughtSignature(id)
	var b strings.Builder
	b.Grow(len(base))
	for _, r := range base {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "tool_use_id"
	}
	return b.String()
}

// parseDataURL splits a base64 data URL into its media type and payload. The
// ;base64, marker is the only spelling the reference reads — a data URL without
// it is refused rather than guessed at — and a JSON-escaped slash in the media
// type (\/) reads as the slash it names.
func parseDataURL(url string) (media, data string, ok bool) {
	rest := strings.TrimPrefix(url, "data:")
	i := strings.Index(rest, ";base64,")
	if i < 0 {
		return "", "", false
	}
	media = strings.ReplaceAll(rest[:i], `\/`, "/")
	return media, rest[i+len(";base64,"):], true
}

// looksBase64 reports whether s is a well-formed base64 payload, the check the
// reference runs before treating a data URL as media rather than text.
func looksBase64(s string) bool {
	if s == "" || len(s)%4 != 0 {
		return false
	}
	_, err := base64.StdEncoding.DecodeString(s)
	return err == nil
}

// isAudioDataURL reports whether a text part's content is exactly a base64
// audio data URL. The reference recognizes that spelling as audio output and
// keeps it out of the text, so the fold must too or a client would read media
// bytes as prose.
func isAudioDataURL(s string) bool {
	if !strings.HasPrefix(s, "data:audio") || !strings.Contains(s, ";base64,") {
		return false
	}
	_, data, ok := parseDataURL(s)
	return ok && looksBase64(data)
}

// uploadedFileURI reports whether uri names a file the target wire already
// holds: an https URI whose path addresses the wire's uploaded-files
// namespace. The reference hands such a URI over instead of fetching it — the
// file answers direct reads with a refusal, so passing it through is the only
// way to use it. The check is on the path's shape rather than on any host:
// this codec carries no host knowledge, and every URI shaped like an upload
// reads the same way here — the same rule the chat-side fold of this wire
// applies.
func uploadedFileURI(uri string) bool {
	rest, ok := strings.CutPrefix(uri, "https://")
	if !ok {
		return false
	}
	if i := strings.IndexAny(rest, "?#"); i >= 0 {
		rest = rest[:i]
	}
	slash := strings.Index(rest, "/")
	if slash < 0 {
		return false
	}
	return strings.HasPrefix(rest[slash:], "/v1beta/files/")
}

// urlMimeTypes maps a URL's file extension onto the MIME type the reference
// infers from it, in the reference's own group order (an extension in two
// groups resolves to the first).
var urlMimeTypes = []struct {
	exts []string
	mime string
}{
	{[]string{".jpg", ".jpeg"}, "image/jpeg"},
	{[]string{".png"}, "image/png"},
	{[]string{".webp"}, "image/webp"},
	{[]string{".mp4"}, "video/mp4"},
	{[]string{".mov"}, "video/mov"},
	{[]string{".mpeg", ".mpg"}, "video/mpeg"},
	{[]string{".avi"}, "video/avi"},
	{[]string{".wmv"}, "video/wmv"},
	{[]string{".mpegps"}, "video/mpegps"},
	{[]string{".flv"}, "video/flv"},
	{[]string{".mp3"}, "audio/mp3"},
	{[]string{".wav"}, "audio/wav"},
	{[]string{".ogg"}, "audio/ogg"},
	{[]string{".pdf"}, "application/pdf"},
	{[]string{".txt"}, "text/plain"},
}

// mimeFromURL infers a MIME type from a URL's extension, ignoring any query
// string. An extension the table does not know reads as none: the reference
// would go fetch the URL to sniff it, which this offline codec cannot do.
func mimeFromURL(url string) string {
	lower := strings.ToLower(url)
	path := lower
	if i := strings.IndexAny(path, "?#"); i >= 0 {
		path = path[:i]
	}
	for _, group := range urlMimeTypes {
		for _, ext := range group.exts {
			if strings.HasSuffix(path, ext) {
				return group.mime
			}
		}
	}
	return ""
}

// roundTripJSON re-encodes a decoded JSON value the way the reference's bridge
// does when it hands arguments from one fold to another: serialize, then parse
// back. Numbers keep the literal they were written with; a value that cannot
// survive the trip rides through unchanged rather than being emptied out.
func roundTripJSON(v any) any {
	b, err := json.Marshal(v)
	if err != nil {
		return v
	}
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return v
	}
	return out
}

// deepCopyValue copies a decoded JSON tree. Maps are memoized by pointer so a
// self-referential schema copies as the shared node it is instead of recursing
// forever.
func deepCopyValue(v any) any {
	return copyValue(v, map[uintptr]any{})
}

func copyValue(v any, seen map[uintptr]any) any {
	switch t := v.(type) {
	case map[string]any:
		ptr := reflect.ValueOf(t).Pointer()
		if got, ok := seen[ptr]; ok {
			return got
		}
		out := make(map[string]any, len(t))
		seen[ptr] = out
		for k, val := range t {
			out[k] = copyValue(val, seen)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = copyValue(val, seen)
		}
		return out
	default:
		return v
	}
}

// targetSchemaKeys is the whole vocabulary the target wire accepts inside a
// tool parameter schema. Every other key is dropped, which is why a $ref has
// to be inlined first: the definition it points at would otherwise be removed
// and leave a dangling pointer.
var targetSchemaKeys = map[string]struct{}{
	"type": {}, "format": {}, "title": {}, "description": {}, "nullable": {},
	"default": {}, "items": {}, "minItems": {}, "maxItems": {}, "enum": {},
	"properties": {}, "propertyOrdering": {}, "required": {}, "minProperties": {},
	"maxProperties": {}, "minimum": {}, "maximum": {}, "minLength": {},
	"maxLength": {}, "pattern": {}, "example": {}, "anyOf": {},
}

// buildTargetSchema coerces a tool parameter schema into the shape the target
// wire accepts: references inlined, a null branch folded into nullable, type
// arrays widened to anyOf, enum fixes, array items completed, an object type
// added where nothing else names one, and every key outside the vocabulary
// dropped. The source tree is copied first — the caller's map is never written
// through, which is this library's pure-function contract.
func buildTargetSchema(v any) (map[string]any, error) {
	out, ok := jsonx.AsMap(v)
	if !ok {
		out = map[string]any{}
	} else {
		out = copyValue(out, map[uintptr]any{}).(map[string]any)
	}
	defs := map[string]any{}
	if d, ok := jsonx.AsMap(out["$defs"]); ok {
		for k, val := range d {
			defs[k] = val
		}
	}
	delete(out, "$defs")
	unpackSchemaRefs(out, defs)
	if err := convertAnyOfNullToNullable(out); err != nil {
		return nil, err
	}
	convertSchemaTypes(out)
	fixEnumEmptyStrings(out)
	fixEnumTypes(out)
	processSchemaItems(out)
	addObjectType(out)
	return filterSchemaFields(out), nil
}

// unpackSchemaRefs walks the whole schema tree and replaces each {"$ref": ...}
// whose last path segment names a definition in scope with a copy of it. The
// scope starts from the caller's defs plus the schema's own definition blocks,
// widens with the definitions each node and each resolved target carry, a
// reference already being resolved on the current path is circular and stays
// as written, and a reference that names nothing stays untouched for the
// target to judge.
func unpackSchemaRefs(schema map[string]any, defs map[string]any) {
	rootDefs := map[string]any{}
	for k, v := range defs {
		rootDefs[k] = v
	}
	for _, key := range []string{"$defs", "definitions"} {
		if d, ok := jsonx.AsMap(schema[key]); ok {
			for k, v := range d {
				rootDefs[k] = v
			}
		}
	}
	type slot struct {
		node   any
		parent any
		key    any
		defs   map[string]any
		chain  map[string]struct{}
	}
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
				childDefs := map[string]any{}
				for k, v := range cur.defs {
					childDefs[k] = v
				}
				if tm, ok := target.(map[string]any); ok {
					for _, key := range []string{"$defs", "definitions"} {
						if d, ok := jsonx.AsMap(tm[key]); ok {
							for k, v := range d {
								childDefs[k] = v
							}
						}
					}
				}
				resolved := deepCopyValue(target)
				replaceSlot(cur.parent, cur.key, resolved)
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
				for _, d := range []map[string]any{nd, nl} {
					for k, v := range d {
						currentDefs[k] = v
					}
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
}

func lastPathSegment(ref string) string {
	for i := len(ref) - 1; i >= 0; i-- {
		if ref[i] == '/' {
			return ref[i+1:]
		}
	}
	return ref
}

// replaceSlot writes a resolved reference back through the container that held
// it.
func replaceSlot(parent, key any, resolved any) {
	switch p := parent.(type) {
	case map[string]any:
		if k, ok := key.(string); ok {
			p[k] = resolved
		}
	case []any:
		if i, ok := key.(int); ok && i >= 0 && i < len(p) {
			p[i] = resolved
		}
	}
}

// convertAnyOfNullToNullable removes a null branch from an anyOf and marks
// every remaining branch nullable, the spelling the target reads for an
// optional value. An anyOf with nothing but null branches names no type at all
// and fails the request rather than silently meaning something else.
func convertAnyOfNullToNullable(schema map[string]any) error {
	anyof, ok := jsonx.AsSlice(schema["anyOf"])
	if ok {
		containsNull := false
		kept := make([]any, 0, len(anyof))
		for _, atype := range anyof {
			if m, ok := jsonx.AsMap(atype); ok {
				if jsonx.GetString(m, "type") == "null" {
					containsNull = true
					continue
				}
				if len(m) == 0 {
					m["type"] = "object"
				}
			}
			kept = append(kept, atype)
		}
		if len(kept) == 0 {
			return errSchema("an anyOf schema with only a null type")
		}
		if containsNull {
			for _, atype := range kept {
				if m, ok := jsonx.AsMap(atype); ok {
					m["nullable"] = true
				}
			}
		}
		schema["anyOf"] = kept
	}
	if properties, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, value := range properties {
			if m, ok := jsonx.AsMap(value); ok {
				if err := convertAnyOfNullToNullable(m); err != nil {
					return err
				}
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		return convertAnyOfNullToNullable(m)
	}
	return nil
}

// convertSchemaTypes widens a type written as a list into anyOf: object and
// array branches take the type-specific fields with them, primitive branches
// keep only the type.
func convertSchemaTypes(schema map[string]any) {
	if tv, ok := schema["type"]; ok {
		if list, ok := tv.([]any); ok && len(list) > 1 {
			typeSpecific := []string{
				"properties", "required", "additionalProperties", "items",
				"minItems", "maxItems", "minProperties", "maxProperties",
			}
			var anyOf []any
			hasObjectOrArray := false
			for _, t := range list {
				name, ok := t.(string)
				if !ok {
					continue
				}
				if name == "null" {
					anyOf = append(anyOf, map[string]any{"type": "null"})
					continue
				}
				if name == "object" || name == "array" {
					hasObjectOrArray = true
					item := map[string]any{"type": name}
					for _, field := range typeSpecific {
						if v, ok := schema[field]; ok {
							item[field] = v
						}
					}
					anyOf = append(anyOf, item)
					continue
				}
				anyOf = append(anyOf, map[string]any{"type": name})
			}
			if hasObjectOrArray {
				for _, field := range typeSpecific {
					delete(schema, field)
				}
			}
			schema["anyOf"] = anyOf
			delete(schema, "type")
		} else if len(list) == 1 {
			schema["type"] = list[0]
		}
	}
	if properties, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, prop := range properties {
			if m, ok := jsonx.AsMap(prop); ok {
				convertSchemaTypes(m)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		convertSchemaTypes(m)
	}
	if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok {
		for _, item := range anyof {
			if m, ok := jsonx.AsMap(item); ok {
				convertSchemaTypes(m)
			}
		}
	}
}

// fixEnumEmptyStrings replaces empty strings in an enum with null: the target
// does not accept empty strings there, and dropping the value entirely would
// change what the enum permits.
func fixEnumEmptyStrings(schema map[string]any) {
	if enum, ok := jsonx.AsSlice(schema["enum"]); ok {
		out := make([]any, 0, len(enum))
		for _, value := range enum {
			if s, ok := value.(string); ok && s == "" {
				out = append(out, nil)
				continue
			}
			out = append(out, value)
		}
		schema["enum"] = out
	}
	if properties, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, value := range properties {
			if m, ok := jsonx.AsMap(value); ok {
				fixEnumEmptyStrings(m)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		fixEnumEmptyStrings(m)
	}
}

// fixEnumTypes removes an enum from a schema whose type is not string: the
// target only allows enums on strings, so the enum is the half that cannot
// ride and the declared type is the caller's word about the value.
func fixEnumTypes(schema map[string]any) {
	if _, has := schema["enum"]; has {
		keep := false
		if t, ok := schema["type"].(string); ok && strings.EqualFold(t, "string") {
			keep = true
		} else if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok {
			for _, item := range anyof {
				if m, ok := jsonx.AsMap(item); ok {
					if t, ok := m["type"].(string); ok && strings.EqualFold(t, "string") {
						keep = true
						break
					}
				}
			}
		}
		if !keep {
			delete(schema, "enum")
		}
	}
	if properties, ok := jsonx.AsMap(schema["properties"]); ok {
		for _, value := range properties {
			if m, ok := jsonx.AsMap(value); ok {
				fixEnumTypes(m)
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		fixEnumTypes(m)
	}
	if anyof, ok := jsonx.AsSlice(schema["anyOf"]); ok {
		for _, item := range anyof {
			if m, ok := jsonx.AsMap(item); ok {
				fixEnumTypes(m)
			}
		}
	}
}

// processSchemaItems completes an array schema's items: the target requires an
// items member whenever the type is array, so a missing or empty one becomes
// the permissive object schema.
func processSchemaItems(schema map[string]any) {
	if t, ok := schema["type"].(string); ok && strings.EqualFold(t, "array") {
		items, has := schema["items"]
		if !has {
			schema["items"] = map[string]any{"type": "object"}
		} else if m, ok := items.(map[string]any); ok && len(m) == 0 {
			schema["items"] = map[string]any{"type": "object"}
		}
	}
	for _, v := range schema {
		if m, ok := jsonx.AsMap(v); ok {
			processSchemaItems(m)
		} else if list, ok := jsonx.AsSlice(v); ok {
			for _, item := range list {
				if m, ok := jsonx.AsMap(item); ok {
					processSchemaItems(m)
				}
			}
		}
	}
}

// addObjectType names a type on every schema that has none and no combinator,
// closes an object that declares no properties, and recurses so nested schemas
// comply too. Function parameters are objects on this wire, whatever the
// caller's schema left unsaid.
func addObjectType(schema map[string]any) {
	if _, hasType := schema["type"]; !hasType {
		_, hasAnyOf := schema["anyOf"]
		_, hasOneOf := schema["oneOf"]
		_, hasAllOf := schema["allOf"]
		if !hasAnyOf && !hasOneOf && !hasAllOf {
			schema["type"] = "object"
		}
	}
	if properties, ok := jsonx.AsMap(schema["properties"]); ok {
		if v, has := schema["required"]; has && v == nil {
			delete(schema, "required")
		}
		if len(properties) == 0 {
			delete(schema, "properties")
			delete(schema, "required")
			schema["type"] = "object"
		} else {
			schema["type"] = "object"
			for _, value := range properties {
				if m, ok := jsonx.AsMap(value); ok {
					addObjectType(m)
				}
			}
		}
	}
	if m, ok := jsonx.AsMap(schema["items"]); ok {
		addObjectType(m)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if values, ok := jsonx.AsSlice(schema[key]); ok {
			for _, value := range values {
				if m, ok := jsonx.AsMap(value); ok {
					addObjectType(m)
				}
			}
		}
	}
}

// filterSchemaFields keeps only the keys the target schema vocabulary defines.
// A schema whose anyOf is present keeps the anyOf alone — the target rejects
// siblings on a combinator — with a title or description moved onto each
// branch so the words are not lost.
func filterSchemaFields(schema map[string]any) map[string]any {
	schema = filterAnyOfFields(schema)
	out := map[string]any{}
	for key, value := range schema {
		if _, ok := targetSchemaKeys[key]; !ok {
			continue
		}
		switch key {
		case "properties":
			if props, ok := jsonx.AsMap(value); ok {
				filtered := map[string]any{}
				for k, v := range props {
					if m, ok := jsonx.AsMap(v); ok {
						filtered[k] = filterSchemaFields(m)
					} else {
						filtered[k] = v
					}
				}
				out[key] = filtered
			}
		case "format":
			if s, ok := value.(string); ok && (s == "enum" || s == "date-time") {
				out[key] = s
			}
		case "items":
			if m, ok := jsonx.AsMap(value); ok {
				out[key] = filterSchemaFields(m)
			} else {
				out[key] = value
			}
		case "anyOf":
			if list, ok := jsonx.AsSlice(value); ok {
				filtered := make([]any, 0, len(list))
				for _, item := range list {
					if m, ok := jsonx.AsMap(item); ok {
						filtered = append(filtered, filterSchemaFields(m))
					} else {
						filtered = append(filtered, item)
					}
				}
				out[key] = filtered
			}
		default:
			out[key] = value
		}
	}
	return out
}

func filterAnyOfFields(schema map[string]any) map[string]any {
	anyof, ok := jsonx.AsSlice(schema["anyOf"])
	if !ok || len(anyof) == 0 {
		return schema
	}
	title := jsonx.GetString(schema, "title")
	description := jsonx.GetString(schema, "description")
	if title != "" || description != "" {
		for _, item := range anyof {
			if m, ok := jsonx.AsMap(item); ok {
				if title != "" {
					m["title"] = title
				}
				if description != "" {
					m["description"] = description
				}
			}
		}
	}
	return map[string]any{"anyOf": anyof}
}

// stripFalseAdditionalProperties removes every additionalProperties:false from
// a tree: the target schema rejects the key, so the reference strips it from
// tool definitions before building. A truthy value (or a schema object) is
// left alone — only the literal false is refused.
func stripFalseAdditionalProperties(v any) {
	switch t := v.(type) {
	case map[string]any:
		if val, ok := t["additionalProperties"]; ok && val == false {
			delete(t, "additionalProperties")
		}
		for _, val := range t {
			stripFalseAdditionalProperties(val)
		}
	case []any:
		for _, item := range t {
			stripFalseAdditionalProperties(item)
		}
	}
}

// strictFreeCopy deep-copies a schema with every strict key removed: a
// structured-output schema rides the target's JSON-schema member, which does
// not read a strictness flag, so the flag is dropped rather than forwarded in
// a shape the wire refuses.
func strictFreeCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			if k == "strict" {
				continue
			}
			out[k] = strictFreeCopy(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = strictFreeCopy(val)
		}
		return out
	default:
		return v
	}
}

// strictifyOutputSchema rewrites a structured-output schema to what the
// reference's strict mode demands before it crosses to the target: every
// object schema that declares properties is closed with
// additionalProperties:false and names all of its properties as required, and
// the combiners and array items are walked so nested objects comply too.
func strictifyOutputSchema(schema map[string]any) {
	if jsonx.GetString(schema, "type") == "object" {
		if props, ok := jsonx.AsMap(schema["properties"]); ok {
			schema["additionalProperties"] = false
			required := make([]any, 0, len(props))
			for k, v := range props {
				required = append(required, k)
				if m, ok := jsonx.AsMap(v); ok {
					strictifyOutputSchema(m)
				}
			}
			schema["required"] = required
		}
	}
	if items, ok := jsonx.AsMap(schema["items"]); ok {
		strictifyOutputSchema(items)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if list, ok := jsonx.AsSlice(schema[key]); ok {
			for _, sub := range list {
				if m, ok := jsonx.AsMap(sub); ok {
					strictifyOutputSchema(m)
				}
			}
		}
	}
	for _, key := range []string{"$defs", "definitions"} {
		if defs, ok := jsonx.AsMap(schema[key]); ok {
			for _, v := range defs {
				if m, ok := jsonx.AsMap(v); ok {
					strictifyOutputSchema(m)
				}
			}
		}
	}
}

// pythonCompactJSON renders a decoded JSON value the way the reference
// serializes tool arguments: ", " and ": " separators, non-ASCII characters
// kept as written, and numbers as their own literals. Map keys are sorted —
// the pipeline reads arguments from maps, which do not carry the document's
// key order, so a single-key argument (the common case) matches exactly and a
// multi-key one is at least deterministic.
func pythonCompactJSON(v any) string {
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
		b.WriteString(t.String())
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
	case float64:
		b.WriteString(pythonFloatText(t))
	default:
		writePythonJSONString(b, jsonx.String(t))
	}
}

// pythonFloatText spells a float the way the reference does: shortest
// round-trip digits and a trailing .0 on a whole number written as a float.
func pythonFloatText(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func writePythonJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	encoded, _ := json.Marshal(s)
	// json.Marshal escapes exactly the two mandatory characters plus the
	// control range, and leaves non-ASCII as written — the reference's
	// spelling for an arguments string.
	body := string(encoded[1 : len(encoded)-1])
	b.WriteString(body)
	b.WriteByte('"')
}

// errSchema reports a schema the target wire refuses outright. The reference
// raises the same class of failure and the request dies there; failing here
// keeps the caller's request from being changed into a different one.
type errSchema string

func (e errSchema) Error() string { return "schema not representable on the target wire: " + string(e) }
