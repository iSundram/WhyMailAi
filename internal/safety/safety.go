package safety

import (
	"regexp"
	"strings"
)

const maxStringLen = 8000

var suspiciousPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)ignore\s+previous\s+instructions`),
	regexp.MustCompile(`(?i)system\s+prompt`),
	regexp.MustCompile(`(?i)jailbreak`),
	regexp.MustCompile(`(?i)<script`),
	regexp.MustCompile(`(?i)169\.254\.169\.254`),
}

// Analyzer detects unsafe patterns and sanitizes user-provided content.
type Analyzer struct{}

// NewAnalyzer creates a safety analyzer.
func NewAnalyzer() *Analyzer { return &Analyzer{} }

// AnalyzeAndSanitize returns safety flags and sanitized input.
func (a *Analyzer) AnalyzeAndSanitize(input map[string]any) ([]string, map[string]any) {
	if input == nil {
		return nil, map[string]any{}
	}
	flags := map[string]struct{}{}
	out := make(map[string]any, len(input))
	for k, v := range input {
		safeVal, valueFlags := sanitizeValue(v)
		for _, f := range valueFlags {
			flags[f] = struct{}{}
		}
		out[k] = safeVal
	}
	final := make([]string, 0, len(flags))
	for k := range flags {
		final = append(final, k)
	}
	return final, out
}

func sanitizeValue(v any) (any, []string) {
	switch x := v.(type) {
	case string:
		return sanitizeString(x)
	case []any:
		flags := map[string]struct{}{}
		out := make([]any, 0, len(x))
		for _, item := range x {
			s, f := sanitizeValue(item)
			for _, one := range f {
				flags[one] = struct{}{}
			}
			out = append(out, s)
		}
		return out, mapKeys(flags)
	case map[string]any:
		flags := map[string]struct{}{}
		out := make(map[string]any, len(x))
		for k, value := range x {
			s, f := sanitizeValue(value)
			for _, one := range f {
				flags[one] = struct{}{}
			}
			out[k] = s
		}
		return out, mapKeys(flags)
	default:
		return v, nil
	}
}

func sanitizeString(s string) (string, []string) {
	flags := map[string]struct{}{}
	clean := strings.Map(func(r rune) rune {
		if r < 32 && r != '\n' && r != '\t' {
			flags["control-characters-sanitized"] = struct{}{}
			return -1
		}
		return r
	}, s)
	for _, p := range suspiciousPatterns {
		if p.MatchString(clean) {
			flags["prompt-injection-suspected"] = struct{}{}
			break
		}
	}
	if len(clean) > maxStringLen {
		clean = clean[:maxStringLen]
		flags["input-truncated"] = struct{}{}
	}
	return clean, mapKeys(flags)
}

func mapKeys(m map[string]struct{}) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// IsActionBlocked indicates if generated action should be blocked for safety reasons.
func IsActionBlocked(action string, flags []string) bool {
	if strings.TrimSpace(action) == "" {
		return false
	}
	for _, f := range flags {
		if f == "prompt-injection-suspected" {
			return true
		}
	}
	return strings.EqualFold(strings.TrimSpace(action), "execute-system-command")
}
