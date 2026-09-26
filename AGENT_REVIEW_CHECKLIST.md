# Agent Review Checklist (Fallen Heroes)

Use this checklist every time an AI agent, Claude session, coding agent, or
automation proposes a change. The agents generate options; the founder
reviews, verifies, prioritizes, and decides. Never merge blindly.

## 1. Mission alignment

- [ ] Does this move Fallen Heroes closer to its core goals?
- [ ] Is the feature solving a real problem?
- [ ] Is it aligned with current roadmap priorities?
- [ ] Is it MVP-friendly?
- [ ] Is it worth maintaining long-term?

## 2. Technical feasibility

- [ ] Can this actually be built with current technology?
- [ ] Is the implementation realistic?
- [ ] Are dependencies available?
- [ ] Is browser compatibility acceptable?
- [ ] Are performance assumptions realistic?
- [ ] Are cost estimates realistic?
- [ ] Are timelines realistic?

## 3. Architecture review

- [ ] Does the design follow existing architecture patterns?
- [ ] Does it introduce unnecessary complexity?
- [ ] Is there a simpler solution?
- [ ] Are interfaces clearly defined?
- [ ] Can it scale?
- [ ] Can multiple developers work on it?
- [ ] Is it modular?

## 4. Code quality

- [ ] Strong typing (TypeScript, or strict JSDoc where there is no build step)?
- [ ] No unnecessary `any`?
- [ ] Clear separation of concerns?
- [ ] Reusable components?
- [ ] Testable design?
- [ ] Low coupling?
- [ ] High cohesion?

## 5. Performance review

WebGPU / graphics

- [ ] GPU-driven where possible?
- [ ] CPU bottlenecks minimized?
- [ ] Memory budget defined?
- [ ] VRAM usage reasonable?
- [ ] Dynamic quality supported?
- [ ] Benchmarking included?
- [ ] Telemetry included?

Browser

- [ ] Chrome tested?
- [ ] Edge tested?
- [ ] Firefox fallback considered?
- [ ] Mobile degradation path exists?

## 6. AI verification

- [ ] Did the agent provide evidence?
- [ ] Did the agent explain reasoning?
- [ ] Any obvious hallucinations?
- [ ] Any made-up APIs?
- [ ] Any unverified assumptions?
- [ ] Any unsupported performance claims?

Red flags

- "This definitely works"
- No benchmarks
- No documentation
- No validation plan
- Unrealistic timelines

## 7. Security review

- [ ] Input validation exists?
- [ ] No secrets in code?
- [ ] Authentication reviewed?
- [ ] Authorization reviewed?
- [ ] Telemetry privacy-safe?
- [ ] Error handling implemented?
- [ ] Rate limits considered?

## 8. Business review

- [ ] Revenue impact?
- [ ] User value?
- [ ] Competitive advantage?
- [ ] Defensible technology?
- [ ] Development cost justified?
- [ ] Maintenance cost justified?

## 9. Graphics review (Aurelius Dynamic)

Visual quality

- [ ] Atmosphere quality reviewed?
- [ ] Lighting quality reviewed?
- [ ] Motion quality reviewed?
- [ ] Material quality reviewed?
- [ ] Camera quality reviewed?

Performance

- [ ] Stable frametimes?
- [ ] Tiered quality system?
- [ ] Benchmark harness integrated?
- [ ] GPU profiling available?
- [ ] Adaptive scaling available?

DARPA test. Ask: does this look like a consumer website or a mission-critical
visualization platform? Desired answer: mission-critical platform.

## 10. Testing requirements

- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance tests
- [ ] Stress tests
- [ ] Browser tests
- [ ] Mobile tests
- [ ] Accessibility tests

## 11. Claude output review

Before accepting code from Claude:

- [ ] Read every file
- [ ] Verify imports
- [ ] Verify package compatibility
- [ ] Check generated types
- [ ] Run linting
- [ ] Run tests
- [ ] Run benchmark
- [ ] Review architecture impact

Never merge blindly.

## 12. Go / No-Go decision

| Verdict | Signals | Action |
| --- | --- | --- |
| GREEN | Technically sound · aligned with roadmap · performance validated · risk acceptable | Implement |
| YELLOW | Useful idea · needs refinement · missing validation | Prototype first |
| RED | Scope creep · high complexity · unclear value · unsupported claims | Reject |

## 13. Merge gate

Every line must be true before a change merges. If one is false the change is
not done, whatever else is finished.

- [ ] Typecheck, lint, and tests pass.
- [ ] Every new import and API exists in the pinned version.
- [ ] Benchmark shows no frame-time regression on the reference device.
- [ ] The visual diff against baseline frames is reviewed.
- [ ] It works on Chrome, Safari, and the no-WebGPU path.
- [ ] This is the simplest version that solves the stated problem.
- [ ] It serves the written mission.
- [ ] You can state how you'll know it succeeded.

## Golden rule

Before implementing any agent output, ask:

1. Is it technically correct?
2. Is it aligned with the mission?
3. Is it scalable?
4. Is it maintainable?
5. Is it worth the complexity?
6. Is there a simpler solution?
7. Can I measure success?

Five or more yes: proceed. Fewer than five: re-evaluate before investing
development time.

Your role as founder is not to accept agent output. Your role is to review,
verify, prioritize, and decide. The agents generate options; you choose the
direction.
