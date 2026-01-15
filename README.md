# Fractals for Computational Thinking

This repository contains an educational software developed as part of an
undergraduate thesis in Computer Science at the Federal University of Pelotas (UFPel).

The project explores the use of fractal geometry as a multidisciplinary tool
to support the teaching and learning of Computational Thinking concepts
in secondary education.

## Project Overview

Computational Thinking is a fundamental skill for the 21st century and is
officially recognized in the Brazilian National Common Core Curriculum (BNCC).
This software was designed to help students understand core Computational
Thinking concepts through interactive visualization and manipulation of fractals.

By combining mathematics, computer science, biology, and art, the tool promotes
a multidisciplinary learning experience focused on abstraction, decomposition,
pattern recognition, algorithms, and parallelism.

## Features

- Interactive visualization of classic fractals
  - Mandelbrot Set
  - Fractal flowers
  - L-Systems
- Real-time parameter manipulation
- Visual exploration of recursive and iterative processes
- Educational focus on Computational Thinking pillars

## Educational Context

This software was applied in classroom activities with secondary school students
as part of a pedagogical intervention described in the undergraduate thesis:

**"Applicability of Fractals in Computational Thinking: A Multidisciplinary Approach"**

The activities were designed to stimulate:
- Abstraction
- Decomposition
- Pattern recognition
- Algorithmic thinking
- Parallel reasoning

## Project Origin and Adaptation

This project is based on the following open-source software:

- **Original Project:** Mandel2Us
- **Author:** Lucas Morais
- **Repository:** https://github.com/lucaszm7/Mandel2Us

The original project provided the technical foundation for fractal rendering.
This repository extends and adapts the software for educational purposes,
with a focus on Computational Thinking and multidisciplinary teaching. The original project served as a technical foundation, and significant modifications
were made to:
- Adapt the software for educational use
- Integrate Computational Thinking concepts
- Align the tool with the Brazilian BNCC guidelines
- Support multidisciplinary classroom activities

All adaptations were carried out exclusively for academic and educational purposes.
The original authorship is fully acknowledged.

## Technologies Used

- Programming Language: C++
- Graphics / Visualization: olcPixelGameEngine

## How to Run

### Linux
```bash
# Example
git clone https://github.com/your-username/fractals-for-computational-thinking.git
make install
make gen
make run
```
### Windows
Using MSYS2 + MinGW-64
```bah
g++ -fopenmp Application.cpp \
    -luser32 -lgdi32 -lopengl32 -lgdiplus -lShlwapi -ldwmapi \
    -lstdc++fs -static -std=c++17 -O3 -mavx2 \
    -o app
```

Alternatively, you can simply run the **app** file.

