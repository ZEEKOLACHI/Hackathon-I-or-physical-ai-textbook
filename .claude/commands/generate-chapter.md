# Generate Chapter Content

Generate comprehensive chapter content for the Physical AI & Humanoid Robotics textbook.

## Usage

```
/generate-chapter <part-number> <chapter-number> <chapter-title>
```

## Arguments

- `part-number`: Part of the book (1-7)
- `chapter-number`: Chapter number within the part
- `chapter-title`: Title of the chapter

## Process

1. **Research Phase**
   - Search for authoritative sources on the topic
   - Identify key concepts, algorithms, and implementations
   - Find relevant code examples and best practices

2. **Structure Phase**
   - Create chapter outline with sections and subsections
   - Map learning objectives to sections
   - Plan code examples and diagrams

3. **Content Generation**
   - Write introduction with learning objectives
   - Generate each section with:
     - Clear explanations of concepts
     - Python/ROS 2 code examples
     - Diagrams or ASCII art where helpful
     - Real-world applications
   - Include chapter summary and exercises

4. **Quality Check**
   - Verify code examples are syntactically correct
   - Ensure difficulty progression is appropriate
   - Check citations and references

## Output Format

```markdown
---
id: ch-{part}-{chapter}
title: {chapter-title}
sidebar_position: {chapter}
difficulty: beginner|intermediate|advanced
estimated_time: {minutes}
prerequisites: [{prerequisite-chapter-ids}]
---

# {Chapter Title}

{Introduction paragraph}

## Learning Objectives

After completing this chapter, you will be able to:
- {objective 1}
- {objective 2}
- {objective 3}

## {Section 1}

{Content with code examples}

```python
# Example code
```

## {Section 2}

{Content continues...}

## Summary

{Chapter summary}

## Exercises

1. {Exercise 1}
2. {Exercise 2}

## Further Reading

- {Reference 1}
- {Reference 2}
```

## Example

```
/generate-chapter 2 5 "Sensor Fusion Fundamentals"
```

This generates a complete chapter on sensor fusion with:
- Kalman filter explanations
- Python implementation examples
- ROS 2 integration code
- Practical exercises
