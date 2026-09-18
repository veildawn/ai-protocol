package protocol

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// This file is the codec's border, written as a test rather than as a
// convention. ai-protocol is a PUBLIC DIALECT CODEC and nothing else: three
// public dialects, request/response/SSE conversion, explicit error semantics.
// It does not speak HTTP, does not know a vendor, and carries no routing,
// billing, account or quota policy. Those rules are only worth anything if a
// change that breaks one fails here, in the repository that would break it —
// see the contract section of README.md.
//
// Deliberately NOT checked: the behavior of the mappings themselves. That is
// what the golden/conformance tests are for.

// moduleGoFiles lists this module's Go sources, skipping tests when asked.
func moduleGoFiles(t *testing.T, includeTests bool) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "testdata", "node_modules":
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		if !includeTests && strings.HasSuffix(path, "_test.go") {
			return nil
		}
		out = append(out, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	sort.Strings(out)
	if len(out) == 0 {
		t.Fatal("no Go files found under the module root; this guard would pass vacuously")
	}
	return out
}

func parseModule(t *testing.T, includeTests bool) (map[string]*ast.File, *token.FileSet) {
	t.Helper()
	fset := token.NewFileSet()
	files := map[string]*ast.File{}
	for _, path := range moduleGoFiles(t, includeTests) {
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		files[filepath.ToSlash(path)] = file
	}
	return files, fset
}

// modulePath reads the module path from go.mod, so the import guard can tell
// the module's own packages from everything else.
func modulePath(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatalf("go.mod: %v", err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "module "); ok {
			return strings.TrimSpace(rest)
		}
	}
	t.Fatal("go.mod declares no module path")
	return ""
}

// importProblems reports every way one import can violate the codec's border.
// It is a pure function so the rules can be tested against synthetic paths —
// an actual third-party import cannot be injected into this module, because
// the module has no dependencies for it to resolve against.
func importProblems(path, importPath, module string) []string {
	if importPath == module || strings.HasPrefix(importPath, module+"/") {
		return nil
	}
	var out []string
	// A third-party import arrives as a path whose first segment contains a
	// dot; the standard library's never do.
	first, _, _ := strings.Cut(importPath, "/")
	if strings.Contains(first, ".") {
		out = append(out, path+" imports "+importPath+": this module has no dependencies beyond the standard library, and the codec must not acquire one")
	}
	if importPath == "net" || strings.HasPrefix(importPath, "net/") {
		out = append(out, path+" imports "+importPath+": the codec does not speak HTTP, and it must not be able to")
	}
	return out
}

// vendorMentions reports the vendor vocabulary a source carries. Pure, so the
// denylist can be proven to bite without editing a real file.
func vendorMentions(src string) []string {
	lower := strings.ToLower(src)
	var out []string
	for _, token := range vendorVocabulary {
		if strings.Contains(lower, token) {
			out = append(out, token)
		}
	}
	return out
}

// The codec is stdlib-only and offline. The dependency statement itself is
// checked first, because an import guard can only see imports this repository
// compiles, while a require directive is the thing a consumer inherits.
func TestCodecIsStdlibOnlyAndOffline(t *testing.T) {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatalf("go.mod: %v", err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "require ") || trimmed == "require (" {
			t.Errorf("go.mod carries a require directive (%q): this module is standard-library only by design — a dependency here is inherited by every consumer of the codec", trimmed)
		}
	}

	module := modulePath(t)
	files, _ := parseModule(t, false)
	for path, file := range files {
		for _, imp := range file.Imports {
			for _, problem := range importProblems(path, strings.Trim(imp.Path.Value, `"`), module) {
				t.Error(problem)
			}
		}
	}
}

// The rules above are only worth having if they bite. An actual third-party
// import cannot be staged here (nothing resolves it), so the predicate is
// driven directly with the paths a violation would use.
func TestImportRulesCatchWhatTheyClaim(t *testing.T) {
	const module = "github.com/veildawn/ai-protocol"
	cases := []struct {
		importPath string
		problems   int
	}{
		{"strings", 0},
		{"encoding/json", 0},
		{module + "/stream", 0},
		{"github.com/veildawn/ai-protocol", 0},
		{"net/http", 1},
		{"net/url", 1},
		{"net", 1},
		{"example.com/evil", 1},
		{"github.com/other/module", 1},
	}
	for _, tc := range cases {
		if got := importProblems("x.go", tc.importPath, module); len(got) != tc.problems {
			t.Errorf("importProblems(%q) = %v, want %d problem(s)", tc.importPath, got, tc.problems)
		}
	}
}

// Exactly three public dialects. A fourth would be a new wire format the codec
// claims to understand — a boundary decision, not a rename.
func TestDialectVocabularyIsClosed(t *testing.T) {
	files, _ := parseModule(t, false)
	file, ok := files["dialect.go"]
	if !ok {
		t.Fatal("dialect.go is missing; this guard is pinned to it")
	}
	got := map[string]string{}
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.CONST {
			continue
		}
		for _, spec := range gen.Specs {
			value, ok := spec.(*ast.ValueSpec)
			if !ok || len(value.Names) != len(value.Values) {
				continue
			}
			declared, ok := value.Type.(*ast.Ident)
			if !ok || declared.Name != "Dialect" {
				continue
			}
			for i, name := range value.Names {
				lit, ok := value.Values[i].(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					t.Errorf("dialect %s is not a plain string literal", name.Name)
					continue
				}
				got[name.Name] = strings.Trim(lit.Value, `"`)
			}
		}
	}
	want := map[string]string{"Chat": "chat", "Messages": "messages", "Responses": "responses"}
	if len(got) != len(want) {
		t.Fatalf("dialect constants are %v, want exactly %v", got, want)
	}
	for name, value := range want {
		if got[name] != value {
			t.Errorf("dialect %s = %q, want %q — the constant name and the wire spelling are one thing here, not two", name, got[name], value)
		}
	}
}

// vendorVocabulary is the knowledge that belongs to a provider plugin, never to
// the codec: brand names, vendor hosts, vendor-only parameter spellings. The
// scan covers non-test sources only, so a fixture may still carry a model id
// like "claude-3" without tripping it.
var vendorVocabulary = []string{
	"openai.com", "anthropic.com", "googleapis.com",
	"gemini", "claude", "gpt", "qwen", "deepseek", "grok", "mistral",
	"cohere", "ollama", "llama", "bedrock", "vertex", "azure",
	"moonshot", "zhipu", "kimi", "litellm",
}

func TestCodecCarriesNoVendorKnowledge(t *testing.T) {
	for _, path := range moduleGoFiles(t, false) {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if tokens := vendorMentions(string(raw)); len(tokens) > 0 {
			t.Errorf("%s mentions %v: vendor knowledge belongs to a provider plugin. If this is a false positive, reword it or narrow vendorVocabulary — do not delete the guard.", path, tokens)
		}
	}
}

func TestVendorRulesCatchWhatTheyClaim(t *testing.T) {
	for _, tc := range []struct {
		src   string
		bites bool
	}{
		{"// converts chat to messages", false},
		{"host := \"OPENAI.COM\"", true},
		{"// claude-3 shaped ids are the caller's business", true},
		{"// gpt", true},
	} {
		if got := len(vendorMentions(tc.src)) > 0; got != tc.bites {
			t.Errorf("vendorMentions(%q) bit=%v, want %v", tc.src, got, tc.bites)
		}
	}
}

// boundarySurfaceAllowlist is the codec's public API, per package, as of the
// v0.3.0 boundary. It is written out rather than derived so that ADDING public
// API is a deliberate act: the failure below tells you to update this list and
// the contract in README.md, which is the review this guard exists to force.
//
// Top-level declarations and exported methods of exported types are covered;
// struct FIELDS are not, so a new field on an exported struct still needs a
// human to notice it.
var boundarySurfaceAllowlist = map[string][]string{
	"chatmessages":      {"ChatResponseToMessages", "ChatToMessages", "MessagesResponseToChat", "MessagesToChat", "PipeChatToMessages", "PipeMessagesToChat", "StreamOpts"},
	"chatresponses":     {"ChatResponseToResponses", "ChatToResponses", "PipeChatToResponses", "PipeResponsesToChat", "ResponsesResponseToChat", "ResponsesToChat", "StreamOpts"},
	"internal/jsonx":    {"AsMap", "AsSlice", "Bool", "CloneMap", "GetString", "Int", "JoinText", "Marshal", "String", "UnmarshalMap"},
	"messagesresponses": {"MessagesResponseToResponses", "MessagesToResponses", "PipeMessagesToResponses", "PipeResponsesToMessages", "ResponsesRequestToMessages", "ResponsesToMessages", "StreamOpts"},
	"protocol": {
		"BodyLosses", "Chat", "ConversionError", "ConversionError.Error", "ConversionError.Unwrap",
		"ConvertOptions", "ConvertRequest", "ConvertRequestWith", "ConvertResponse", "ConvertResponseWith",
		"Converted", "Dialect", "Dialect.String", "Loss", "LossFileReference", "LossLogitBias", "LossLogprobs",
		"LossMultiChoice", "LossPenaltyControls", "LossSeed", "LossSet", "LossSet.Add", "LossSet.Union",
		"LossStructuredOutput", "LossToolArguments", "LossUnknownContent", "Messages",
		"MissingRequiredFieldError", "MissingRequiredFieldError.Error", "Normalize",
		"PipeStream", "PipeStreamWith", "Responses", "StaticLosses", "StreamOpts", "StreamResult",
		"UnsupportedParamError", "UnsupportedParamError.Error",
	},
	"stream": {
		"Chat", "Dialect", "Event", "Event.Dialect", "Event.Encode", "Fail", "InBandError", "IsDone",
		"Messages", "Observer", "Observer.Feed", "Responses", "Result", "Scan", "WriteDone",
		"WriteEvent", "WriteEventWith",
	},
	"types": {
		"BudgetFromEffort", "BudgetHigh", "BudgetLow", "BudgetMax", "BudgetMedium", "BudgetMinimal",
		"BudgetXHigh", "EffortFromBudget", "MinThinking",
	},
}

func actualSurface(files map[string]*ast.File) map[string][]string {
	out := map[string][]string{}
	for path, file := range files {
		pkg := filepath.ToSlash(filepath.Dir(path))
		if pkg == "." {
			pkg = file.Name.Name
		}
		add := func(name string) {
			if ast.IsExported(name) {
				out[pkg] = append(out[pkg], name)
			}
		}
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				name := d.Name.Name
				if d.Recv != nil && len(d.Recv.List) > 0 {
					if recv := receiverName(d.Recv.List[0].Type); recv != "" {
						name = recv + "." + name
					}
				}
				add(name)
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						add(s.Name.Name)
					case *ast.ValueSpec:
						for _, n := range s.Names {
							add(n.Name)
						}
					}
				}
			}
		}
	}
	for pkg := range out {
		sort.Strings(out[pkg])
	}
	return out
}

func receiverName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.StarExpr:
		return receiverName(t.X)
	case *ast.Ident:
		return t.Name
	default:
		return ""
	}
}

// Public API is a promise to every consumer, and this library's promise is
// narrow on purpose. Growing it is allowed — it just has to be a decision
// somebody makes, which is what updating this list means.
func TestExportedSurfaceIsUnchanged(t *testing.T) {
	files, _ := parseModule(t, false)
	got := actualSurface(files)
	checked := 0
	for pkg, want := range boundarySurfaceAllowlist {
		have := got[pkg]
		checked++
		added := difference(have, want)
		removed := difference(want, have)
		if len(added) == 0 && len(removed) == 0 {
			continue
		}
		t.Errorf("package %s: the exported surface moved.\n  added:   %v\n  removed: %v\n"+
			"Public API is the codec's contract. If the addition is genuinely public-dialect conversion — no HTTP, no vendor, no routing/billing/account/quota policy — update boundarySurfaceAllowlist and the contract in README.md. If it is not, it belongs in the consumer.", pkg, added, removed)
	}
	for pkg := range got {
		if _, ok := boundarySurfaceAllowlist[pkg]; !ok {
			t.Errorf("package %s is not in boundarySurfaceAllowlist; add it (new packages get the same review)", pkg)
		}
	}
	if checked != len(got) {
		t.Fatalf("checked %d packages, module has %d", checked, len(got))
	}
}

func difference(a, b []string) []string {
	in := make(map[string]bool, len(b))
	for _, v := range b {
		in[v] = true
	}
	var out []string
	for _, v := range a {
		if !in[v] {
			out = append(out, v)
		}
	}
	return out
}
