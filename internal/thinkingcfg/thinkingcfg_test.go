package thinkingcfg

import "testing"

// The live failure this pins (2026-10-28): a Gemini client asked for
// thinkingLevel HIGH, no reader knew the spelling, the absent budget bucketed as
// the shallowest rung, and the request SUCCEEDED at minimum depth on a
// 101,780-token conversation.
func TestEffortReadsThinkingLevel(t *testing.T) {
	for _, tc := range []struct {
		level string
		want  string
	}{
		{"HIGH", "high"},
		{"high", "high"},
		{"HIGH", "high"},
		{"LOW", "low"},
		{"MEDIUM", "medium"},
		{"MINIMAL", "minimal"},
		{" high ", "high"},
	} {
		t.Run(tc.level, func(t *testing.T) {
			got, ok := Effort(map[string]any{"thinkingLevel": tc.level, "includeThoughts": true})
			if !ok {
				t.Fatal("a stated level answered no effort")
			}
			if got != tc.want {
				t.Fatalf("effort = %q, want %q", got, tc.want)
			}
		})
	}
}

// The two spellings must agree: a level resolves through the same budget scale
// the numeric form does, or the request and the usage row would disagree about
// how deep the model was asked to think.
func TestEffortLevelAndBudgetAgree(t *testing.T) {
	pairs := []struct {
		level  string
		budget int
	}{
		{"minimal", 128}, {"low", 1024}, {"medium", 2048}, {"high", 4096},
	}
	for _, p := range pairs {
		fromLevel, ok := Effort(map[string]any{"thinkingLevel": p.level})
		if !ok {
			t.Fatalf("%s: no effort", p.level)
		}
		fromBudget, ok := Effort(map[string]any{"thinkingBudget": p.budget})
		if !ok {
			t.Fatalf("%d: no effort", p.budget)
		}
		if fromLevel != fromBudget {
			t.Fatalf("%s and %d disagree: %q vs %q", p.level, p.budget, fromLevel, fromBudget)
		}
	}
}

// Switched-off thinking is "none" whatever else the config says: a disabled
// block that also names a budget still reasons not at all.
func TestEffortDisabledWinsOverEveryDepth(t *testing.T) {
	for _, tc := range []map[string]any{
		{"includeThoughts": false},
		{"includeThoughts": false, "thinkingBudget": 4096},
		{"includeThoughts": false, "thinkingLevel": "HIGH"},
	} {
		got, ok := Effort(tc)
		if !ok || got != "none" {
			t.Fatalf("%v answered %q/%v, want none", tc, got, ok)
		}
	}
}

// An explicit budget outranks a level: it is the number the client chose.
func TestEffortBudgetBeatsLevel(t *testing.T) {
	got, ok := Effort(map[string]any{"thinkingLevel": "HIGH", "thinkingBudget": 1024})
	if !ok {
		t.Fatal("no effort")
	}
	if got != "low" {
		t.Fatalf("effort = %q, want low (the stated 1024 budget)", got)
	}
}

// A config that names no depth answers false, NOT the lowest rung. "Thinking on,
// depth unspecified" is a request for the target's default; silently handing
// back "minimal" would be this codec substituting its own choice for the
// client's.
func TestEffortStatesNothingWhenNoDepthIsStated(t *testing.T) {
	for _, tc := range []map[string]any{
		nil,
		{},
		{"includeThoughts": true},
		{"thinkingLevel": "galactic"},
	} {
		if got, ok := Effort(tc); ok {
			t.Fatalf("%v answered %q, want nothing stated", tc, got)
		}
	}
}

// A level nobody recognises is not guessed at: the wire may name a depth this
// codec has never seen, and inventing a budget would claim depth the client
// never asked for.
func TestBudgetIgnoresUnknownLevels(t *testing.T) {
	if n, ok := Budget(map[string]any{"thinkingLevel": "galactic"}); ok {
		t.Fatalf("an unknown level resolved to %d", n)
	}
}

// Zero is a stated budget, not an absence: "do not think" is a decision.
func TestBudgetZeroIsStated(t *testing.T) {
	n, ok := Budget(map[string]any{"thinkingBudget": 0})
	if !ok || n != 0 {
		t.Fatalf("zero budget answered %d/%v, want 0/true", n, ok)
	}
}
