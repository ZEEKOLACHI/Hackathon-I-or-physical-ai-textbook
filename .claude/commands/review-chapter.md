# Review Chapter

Review a chapter for technical accuracy, clarity, and completeness.

## Usage

```
/review-chapter <chapter-id>
```

## Review Checklist

### Content Quality
- [ ] Learning objectives are clear and measurable
- [ ] Concepts are explained progressively (simple to complex)
- [ ] Technical accuracy verified
- [ ] Code examples are correct and runnable
- [ ] Diagrams/illustrations support understanding

### Structure
- [ ] Introduction sets context and motivation
- [ ] Sections flow logically
- [ ] Appropriate use of headings (h2, h3, h4)
- [ ] Summary captures key points
- [ ] Exercises reinforce learning

### Code Examples
- [ ] Syntax is correct (Python 3.11+)
- [ ] Comments explain the "why" not just the "what"
- [ ] Error handling is appropriate
- [ ] ROS 2 patterns follow best practices
- [ ] Examples are self-contained or dependencies noted

### Accessibility
- [ ] Language is clear and concise
- [ ] Jargon is explained on first use
- [ ] Alt text for images (if any)
- [ ] Code blocks have language specified

### Metadata
- [ ] Frontmatter is complete
- [ ] Difficulty level is accurate
- [ ] Prerequisites are listed
- [ ] Estimated time is reasonable

## Output Format

```markdown
## Chapter Review: {chapter-title}

### Summary
{Brief overview of chapter content}

### Strengths
- {strength 1}
- {strength 2}

### Issues Found
1. **[Severity]** {issue description}
   - Location: {section/line}
   - Suggestion: {how to fix}

### Recommendations
- {recommendation 1}
- {recommendation 2}

### Score: {X}/10
```
