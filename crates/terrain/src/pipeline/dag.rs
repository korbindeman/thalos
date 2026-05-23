//! [`FieldDag`] — the automatically derived field evaluation order.
//!
//! Fields form a directed acyclic graph: when field A's expression references
//! field B, there is an edge B → A (B must be evaluated first). The
//! topological order is computed from the expression contents — the author
//! never specifies field order manually (spec §3). Cycles, references to
//! unknown fields, and duplicate field names are rejected at build time with a
//! clear error.

use std::collections::HashMap;

use crate::pipeline::field::Field;

/// Why building a [`FieldDag`] failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagError {
    /// Two fields share a name; names must be unique within a planet.
    DuplicateField { name: String },
    /// A field's expression references a field that does not exist.
    UnknownReference {
        /// The field whose expression contains the dangling reference.
        field: String,
        /// The name it referenced.
        referenced: String,
    },
    /// The reference graph contains a cycle. Lists the fields that could not
    /// be ordered (every member is part of, or downstream of, a cycle).
    Cycle { fields: Vec<String> },
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateField { name } => {
                write!(f, "duplicate field name `{name}`")
            }
            Self::UnknownReference { field, referenced } => {
                write!(f, "field `{field}` references unknown field `{referenced}`")
            }
            Self::Cycle { fields } => {
                write!(f, "field reference cycle among: {}", fields.join(", "))
            }
        }
    }
}

impl std::error::Error for DagError {}

/// A validated topological evaluation order over a field bag.
#[derive(Debug, Clone)]
pub struct FieldDag {
    /// Field indices in dependency-first order: every field appears after all
    /// fields it references.
    order: Vec<usize>,
}

impl FieldDag {
    /// Build the evaluation order for `fields`, or report why it can't.
    pub fn build(fields: &[Field]) -> Result<Self, DagError> {
        // Name → index, rejecting duplicates.
        let mut index: HashMap<&str, usize> = HashMap::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            if index.insert(field.name.as_str(), i).is_some() {
                return Err(DagError::DuplicateField {
                    name: field.name.clone(),
                });
            }
        }

        // Dependencies (deduplicated) per field, validating references.
        let mut deps: Vec<Vec<usize>> = Vec::with_capacity(fields.len());
        for field in fields {
            let mut field_deps = Vec::new();
            for referenced in field.dependencies() {
                let Some(&dep) = index.get(referenced.as_str()) else {
                    return Err(DagError::UnknownReference {
                        field: field.name.clone(),
                        referenced,
                    });
                };
                field_deps.push(dep);
            }
            deps.push(field_deps);
        }

        // Kahn's algorithm. in_degree[f] = number of fields f depends on;
        // dependents[d] = fields that depend on d.
        let mut in_degree = vec![0usize; fields.len()];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); fields.len()];
        for (f, field_deps) in deps.iter().enumerate() {
            in_degree[f] = field_deps.len();
            for &d in field_deps {
                dependents[d].push(f);
            }
        }

        let mut ready: Vec<usize> = (0..fields.len()).filter(|&f| in_degree[f] == 0).collect();
        let mut order = Vec::with_capacity(fields.len());
        while let Some(f) = ready.pop() {
            order.push(f);
            for &dependent in &dependents[f] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    ready.push(dependent);
                }
            }
        }

        if order.len() != fields.len() {
            // Remaining nodes (in_degree > 0) are tangled in or below a cycle.
            let mut cyclic: Vec<String> = (0..fields.len())
                .filter(|&f| in_degree[f] > 0)
                .map(|f| fields[f].name.clone())
                .collect();
            cyclic.sort();
            return Err(DagError::Cycle { fields: cyclic });
        }

        Ok(Self { order })
    }

    /// Field indices in dependency-first evaluation order.
    pub fn order(&self) -> &[usize] {
        &self.order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::expr::Expr;

    fn f(name: &str, expr: Expr) -> Field {
        Field::scalar(name, expr)
    }

    /// For every field, all of its referenced fields must appear earlier.
    fn assert_topologically_ordered(fields: &[Field], dag: &FieldDag) {
        let position: HashMap<&str, usize> = dag
            .order()
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (fields[idx].name.as_str(), pos))
            .collect();
        for field in fields {
            let here = position[field.name.as_str()];
            for referenced in field.expr.field_refs() {
                assert!(
                    position[referenced.as_str()] < here,
                    "`{}` must be ordered after its dependency `{}`",
                    field.name,
                    referenced
                );
            }
        }
    }

    #[test]
    fn orders_dependencies_before_dependents() {
        // c = b + 1, b = a * 2, a = const. Order must be a, b, c.
        let fields = vec![
            f("c", Expr::Add(vec![Expr::field("b"), Expr::Const(1.0)])),
            f(
                "b",
                Expr::Scale {
                    x: Box::new(Expr::field("a")),
                    factor: 2.0,
                },
            ),
            f("a", Expr::Const(3.0)),
        ];
        let dag = FieldDag::build(&fields).expect("acyclic");
        assert_topologically_ordered(&fields, &dag);
    }

    #[test]
    fn rejects_cycles() {
        // a -> b -> a.
        let fields = vec![f("a", Expr::field("b")), f("b", Expr::field("a"))];
        match FieldDag::build(&fields) {
            Err(DagError::Cycle { fields }) => {
                assert_eq!(fields, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected cycle error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_reference() {
        let fields = vec![f("a", Expr::field("ghost"))];
        match FieldDag::build(&fields) {
            Err(DagError::UnknownReference { field, referenced }) => {
                assert_eq!(field, "a");
                assert_eq!(referenced, "ghost");
            }
            other => panic!("expected unknown-reference error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_field_names() {
        let fields = vec![f("a", Expr::Const(1.0)), f("a", Expr::Const(2.0))];
        match FieldDag::build(&fields) {
            Err(DagError::DuplicateField { name }) => assert_eq!(name, "a"),
            other => panic!("expected duplicate-field error, got {other:?}"),
        }
    }

    #[test]
    fn self_reference_is_a_cycle() {
        let fields = vec![f("a", Expr::Add(vec![Expr::field("a"), Expr::Const(1.0)]))];
        assert!(matches!(
            FieldDag::build(&fields),
            Err(DagError::Cycle { .. })
        ));
    }
}
