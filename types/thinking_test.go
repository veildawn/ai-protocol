package types

import "testing"

func TestEffortFromBudget(t *testing.T) {
	cases := []struct {
		b int
		e string
	}{
		{0, "minimal"},
		{1023, "minimal"},
		{1024, "low"},
		{2048, "medium"},
		{4096, "high"},
		{5000, "high"},
	}
	for _, c := range cases {
		if got := EffortFromBudget(c.b); got != c.e {
			t.Fatalf("budget %d: %s want %s", c.b, got, c.e)
		}
	}
}

func TestBudgetFromEffort(t *testing.T) {
	b, ok := BudgetFromEffort("minimal")
	if !ok || b != 1024 {
		t.Fatalf("%d %v", b, ok)
	}
	_, ok = BudgetFromEffort("nope")
	if ok {
		t.Fatal("expected fail")
	}
}
