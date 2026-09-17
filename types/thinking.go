package types

// Thinking budget thresholds from LiteLLM v1.100.1 constants.py
// (DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET).
const (
	BudgetMinimal = 128
	BudgetLow     = 1024
	BudgetMedium  = 2048
	BudgetHigh    = 4096
	BudgetXHigh   = 8192
	BudgetMax     = 16384
	MinThinking   = 1024
)

// EffortFromBudget buckets Anthropic thinking.budget_tokens into an OpenAI-style
// reasoning_effort label. LiteLLM: reasoning_effort_from_thinking_budget.
func EffortFromBudget(budget int) string {
	if budget >= BudgetHigh {
		return "high"
	}
	if budget >= BudgetMedium {
		return "medium"
	}
	if budget >= BudgetLow {
		return "low"
	}
	return "minimal"
}

// BudgetFromEffort maps OpenAI reasoning_effort to Anthropic budget_tokens.
func BudgetFromEffort(effort string) (budget int, ok bool) {
	switch effort {
	case "", "none":
		return 0, true
	case "minimal":
		return MinThinking, true // max(128, 1024)
	case "low":
		return BudgetLow, true
	case "medium", "adaptive":
		return BudgetMedium, true
	case "high":
		return BudgetHigh, true
	case "xhigh":
		return BudgetXHigh, true
	case "max":
		return BudgetMax, true
	default:
		return 0, false
	}
}
