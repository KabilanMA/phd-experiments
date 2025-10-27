# Stella

**Version:** 0.1.0.0  
**Author:** Kabilan Mahathevan  
**License:** BSD3  

Stella is a **sparse tensor compiler** written in Haskell. It provides a domain-specific language (DSL) for defining tensor operations, and compiles them efficiently to executable code. The library is designed to handle **high-dimensional sparse tensors** while providing a flexible and composable API for tensor computations.

---

<!-- ## Features -->

<!-- - **DSL for tensor operations** – Abstract tensor computations as high-level expressions.
- **Sparse tensor support** – Efficient handling of sparse tensors with customizable formats.
- **Parser and AST** – Full parsing support using `megaparsec` and structured abstract syntax trees (AST).
- **Executable generation** – Compile tensor operations into optimized code with `stella-exe`.
- **Extensible design** – Easily extendable with new tensor operations or backends. -->

<!-- --- -->

<!-- ## Project Structure

stella/
├── app/
│ └── Main.hs # Entry point for executable
├── src/
│ ├── DSL/
│ │ ├── AST.hs # Abstract Syntax Tree for tensor DSL
│ │ └── Parser.hs # Parser for tensor DSL using megaparsec
├── package.yaml # Project metadata and dependencies
├── stack.yaml # Stack resolver and GHC version
├── README.md # Project documentation
└── .gitignore -->

## Dependencies

The project uses the following Haskell libraries:

- `base` ≥4.18 && <5  
- `containers` ≥0.6 && <0.7  
- `text` ≥2.0 && <2.1  
- `megaparsec` ≥9.5 && <10  
- `mtl` ≥2.3 && <2.4  
- `transformers` ≥0.6 && <0.7  
- `prettyprinter` ≥1.7 && <1.8  
- `directory` ≥1.3 && <1.4  
- `process` ≥1.6 && <1.8  

All dependencies are managed automatically via **Stack**.

## Getting Started

### Prerequisites

- [GHC](https://www.haskell.org/ghc/) (via Stack)
- [Stack](https://docs.haskellstack.org/en/stable/README/)

---

### Build Instructions

1. Clone the repository
2. Initialize Stack (if not already done):
```bash
stack setup
```
3. Build the project
```bash
stack build
```
4. Run the executable:
```bash
stack run stella-exe
```