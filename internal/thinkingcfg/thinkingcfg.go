// Package thinkingcfg reads a Gemini thinkingConfig into the answers the other
// dialects need: the effort label they ask with, and the numeric budget they
// measure against.
//
// It exists because that reading was written out three times — in geminichat,
// geminiresponses and geminimessages — and the copies had already drifted: two
// answered "no depth stated" as the lowest rung and one answered "nothing
// stated" as nothing, so the same config converted differently depending on
// which dialect the client happened to speak.
//
// The wire states depth TWO ways, and this package is the only place that has to
// know both:
//
//   - thinkingBudget, the older spelling: a number of thinking tokens.
//   - thinkingLevel, the Gemini 3 spelling: a named depth.
//
// The level was the one nobody read. A client that asked for thinkingLevel HIGH
// reached a fold with no budget at all, the absent number bucketed as the
// shallowest rung, and the request SUCCEEDED at minimum depth — a failure with
// no error attached to it, which is the worst kind. Observed 2026-10-28: a
// 101,780-token request streamed 18 tokens of placeholder prose and was billed
// in full.
//
// A level resolves through the codec's OWN budget scale (types.Budget*), which
// is what EffortFromBudget buckets, so the two spellings agree by construction
// and the level a client names is the level the usage row records. Writing a
// budget from some other scale would make the request and the bill disagree.
package thinkingcfg

import (
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

// LevelBudgets maps a Gemini 3 thinkingLevel onto the codec's budget scale.
//
// Matching is case-insensitive: Google documents the enum upper-case and its
// SDKs send either spelling. A level that is not here is NOT guessed at — see
// Effort.
var LevelBudgets = map[string]int{
	"minimal": types.BudgetMinimal,
	"low":     types.BudgetLow,
	"medium":  types.BudgetMedium,
	"high":    types.BudgetHigh,
	"xhigh":   types.BudgetXHigh,
	"max":     types.BudgetMax,
}

// LevelBudget answers the budget a named level corresponds to, and whether the
// level is one this codec understands.
func LevelBudget(level string) (int, bool) {
	budget, ok := LevelBudgets[strings.ToLower(strings.TrimSpace(level))]
	return budget, ok
}

// Effort reads a thinkingConfig into the effort label a chat or responses
// request asks with.
//
// The order is deliberate, most specific first:
//
//  1. Thoughts switched off is "none", whatever else the config says — a
//     disabled block that also names a budget still reasons not at all.
//  2. An explicit budget is a number the client chose, so it wins over a level.
//  3. A level, when the config states one.
//  4. Otherwise the config states no depth. That answers false rather than the
//     lowest rung: "thinking on, depth unspecified" is a request for the
//     TARGET's default, and turning it into "minimal" silently hands back the
//     shallowest reasoning as a substitute for the client's own default.
func Effort(tc map[string]any) (string, bool) {
	if len(tc) == 0 {
		return "", false
	}
	if inc, ok := jsonx.Bool(tc["includeThoughts"]); ok && !inc {
		return "none", true
	}
	if budget, ok := Budget(tc); ok {
		return types.EffortFromBudget(budget), true
	}
	return "", false
}

// Budget reads the numeric thinking budget a config states, resolving a level
// onto the codec's scale when the config names one instead.
//
// A config that states neither answers false. That is not the same as zero:
// zero is a client asking for no thinking, which Effort reads as "minimal".
func Budget(tc map[string]any) (int, bool) {
	if len(tc) == 0 {
		return 0, false
	}
	if n, ok := jsonx.Int(tc["thinkingBudget"]); ok {
		return n, true
	}
	if level := jsonx.GetString(tc, "thinkingLevel"); level != "" {
		if n, ok := LevelBudget(level); ok {
			return n, true
		}
	}
	return 0, false
}
