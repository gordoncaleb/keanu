//! Sequences (plates): repeating a piece of model once per item, with proxies linking each item to the previous
//! one. The equivalent of Java's `SequenceBuilder`, `SequenceItem` and `Sequence`.
//!
//! Nodes added to an item are labelled `[name.]Sequence_Item_<index>.<identifier>.<label>`, where the identifier
//! numbers the sequences built with one [`GraphBuilder`] (1, 2, ...). A proxy for label `x` (labelled
//! `proxy_for.x` within the item) is connected to the node labelled `x` in the previous item, or in the initial
//! state for the first item, unless a transition mapping says otherwise.
//!
//! ```
//! use keanu_graph::GraphBuilder;
//! use keanu_graph::sequence::SequenceBuilder;
//!
//! let b = GraphBuilder::new();
//! let start = b.scalar(0.0);
//! let sequence = SequenceBuilder::new(&b)
//!     .with_initial_state("x", start)
//!     .count(3, |item| {
//!         let previous = item.double_proxy_for("x");
//!         let x = b.gaussian(previous, 1.0);
//!         item.add("x", x);
//!     })?;
//! assert_eq!(sequence.len(), 3);
//! let last_x = sequence.items()[2].get("x").unwrap();
//! let (graph, _) = b.build()?;
//! assert_eq!(graph.node(last_x).label(), Some("Sequence_Item_2.1.x"));
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```

use crate::builder::GraphBuilder;
use crate::error::{GraphError, Result};
use crate::graph::NodeId;
use crate::value::DType;
use crate::vars::{BoolVar, DoubleVar, IntVar, Var};
use std::collections::HashMap;

/// The namespace marking a proxy's label.
pub const PROXY_LABEL_MARKER: &str = "proxy_for";

/// The prefix of each item's namespace.
pub const ITEM_PREFIX: &str = "Sequence_Item_";

/// The label of the proxy standing for `label` from the previous item.
pub fn proxy_label_for(label: &str) -> String {
    format!("{PROXY_LABEL_MARKER}.{label}")
}

fn is_item_namespace(namespace: &str) -> bool {
    namespace
        .strip_prefix(ITEM_PREFIX)
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Builds a [`Sequence`] of items in a [`GraphBuilder`].
#[derive(Debug)]
pub struct SequenceBuilder<'b> {
    builder: &'b GraphBuilder,
    name: Option<String>,
    initial_state: HashMap<String, NodeId>,
    transitions: HashMap<String, String>,
}

impl<'b> SequenceBuilder<'b> {
    pub fn new(builder: &'b GraphBuilder) -> Self {
        Self {
            builder,
            name: None,
            initial_state: HashMap::new(),
            transitions: HashMap::new(),
        }
    }

    /// Names the sequence, which becomes the outermost namespace of its labels.
    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.to_owned());
        self
    }

    /// A node the first item's proxy for `label` connects to.
    pub fn with_initial_state<V: Var<'b>>(mut self, label: &str, var: V) -> Self {
        self.initial_state.insert(label.to_owned(), var.id());
        self
    }

    /// Connects proxies labelled `proxy_label` (e.g. `proxy_for.x`) to the previous item's node labelled
    /// `parent_label`, instead of the default (the label without `proxy_for.`).
    pub fn with_transition(mut self, proxy_label: &str, parent_label: &str) -> Self {
        self.transitions
            .insert(proxy_label.to_owned(), parent_label.to_owned());
        self
    }

    /// Builds `count` items, calling `factory` to fill each.
    pub fn count(
        self,
        count: usize,
        mut factory: impl FnMut(&mut SequenceItem<'b>),
    ) -> Result<Sequence> {
        self.build((0..count).map(|_| ()), |item, ()| factory(item))
    }

    /// Builds one item per element of `data`, calling `factory` with each element.
    pub fn from_iter<T>(
        self,
        data: impl IntoIterator<Item = T>,
        factory: impl FnMut(&mut SequenceItem<'b>, T),
    ) -> Result<Sequence> {
        self.build(data, factory)
    }

    fn build<T>(
        self,
        data: impl IntoIterator<Item = T>,
        mut factory: impl FnMut(&mut SequenceItem<'b>, T),
    ) -> Result<Sequence> {
        let identifier = self.builder.next_sequence_identifier();
        let mut items: Vec<Item> = Vec::new();
        for (index, element) in data.into_iter().enumerate() {
            let mut item = SequenceItem {
                builder: self.builder,
                index,
                identifier,
                name: self.name.clone(),
                contents: Vec::new(),
                proxies: Vec::new(),
            };
            factory(&mut item, element);
            if let Some(error) = self.builder.error() {
                return Err(error);
            }
            let previous: &HashMap<String, NodeId> = match items.last() {
                Some(last) => &last.contents,
                None => &self.initial_state,
            };
            for (proxy_label, proxy) in &item.proxies {
                let parent_label = match self.transitions.get(proxy_label) {
                    Some(parent) => parent.clone(),
                    None => proxy_label
                        .strip_prefix(&format!("{PROXY_LABEL_MARKER}."))
                        .ok_or_else(|| {
                            GraphError::Sequence(format!("no transition mapping for {proxy_label}"))
                        })?
                        .to_owned(),
                };
                let parent = *previous.get(&parent_label).ok_or_else(|| {
                    GraphError::Sequence(format!(
                        "cannot find label {parent_label} for {proxy_label} in item {index}"
                    ))
                })?;
                self.builder.connect_ids(*proxy, parent)?;
            }
            items.push(Item {
                index,
                contents: item
                    .contents
                    .iter()
                    .map(|(label, _, id)| (label.clone(), *id))
                    .collect(),
                scoped: item
                    .contents
                    .into_iter()
                    .map(|(_, scoped, id)| (scoped, id))
                    .collect(),
            });
        }
        Ok(Sequence {
            identifier,
            name: self.name,
            items,
        })
    }
}

/// An item being built: the nodes added to it are labelled within the item's namespace.
#[derive(Debug)]
pub struct SequenceItem<'b> {
    builder: &'b GraphBuilder,
    index: usize,
    identifier: usize,
    name: Option<String>,
    /// (unscoped label, scoped label, node)
    contents: Vec<(String, String, NodeId)>,
    /// (unscoped proxy label, node)
    proxies: Vec<(String, NodeId)>,
}

impl<'b> SequenceItem<'b> {
    /// The position of the item in the sequence.
    pub fn index(&self) -> usize {
        self.index
    }

    fn scoped(&self, label: &str) -> String {
        let scoped = format!("{ITEM_PREFIX}{}.{}.{label}", self.index, self.identifier);
        match &self.name {
            Some(name) => format!("{name}.{scoped}"),
            None => scoped,
        }
    }

    fn add_id(&mut self, label: &str, id: NodeId) {
        let outer = label.split('.').next().unwrap_or("");
        if is_item_namespace(outer) {
            self.builder.record(GraphError::Sequence(format!(
                "{label} has already been added to a sequence item"
            )));
            return;
        }
        if self.contents.iter().any(|(l, _, _)| l == label) {
            self.builder.record(GraphError::Sequence(format!(
                "{label} has already been added to item {}",
                self.index
            )));
            return;
        }
        let scoped = self.scoped(label);
        self.builder.set_label_of(id, &scoped);
        self.contents.push((label.to_owned(), scoped, id));
    }

    /// Adds a node to the item under `label`, which becomes its (scoped) label.
    pub fn add<V: Var<'b>>(&mut self, label: &str, var: V) -> V {
        self.add_id(label, var.id());
        var
    }

    /// The node added under `label` (unscoped, or the full scoped label).
    pub fn get(&self, label: &str) -> Option<NodeId> {
        self.contents
            .iter()
            .find(|(l, s, _)| l == label || s == label)
            .map(|(_, _, id)| *id)
    }

    fn proxy_for(&mut self, label: &str, dtype: DType, shape: &[usize]) -> NodeId {
        let id = self.builder.add_proxy(dtype, shape);
        let proxy_label = proxy_label_for(label);
        self.add_id(&proxy_label, id);
        self.proxies.push((proxy_label, id));
        id
    }

    /// A scalar proxy for the node labelled `label` in the previous item.
    pub fn double_proxy_for(&mut self, label: &str) -> DoubleVar<'b> {
        self.double_proxy_for_shape(label, &[])
    }

    /// A proxy of the given shape for the node labelled `label` in the previous item.
    pub fn double_proxy_for_shape(&mut self, label: &str, shape: &[usize]) -> DoubleVar<'b> {
        DoubleVar::wrap(self.builder, self.proxy_for(label, DType::F64, shape))
    }

    pub fn int_proxy_for(&mut self, label: &str) -> IntVar<'b> {
        self.int_proxy_for_shape(label, &[])
    }

    pub fn int_proxy_for_shape(&mut self, label: &str, shape: &[usize]) -> IntVar<'b> {
        IntVar::wrap(self.builder, self.proxy_for(label, DType::I64, shape))
    }

    pub fn bool_proxy_for(&mut self, label: &str) -> BoolVar<'b> {
        self.bool_proxy_for_shape(label, &[])
    }

    pub fn bool_proxy_for_shape(&mut self, label: &str, shape: &[usize]) -> BoolVar<'b> {
        BoolVar::wrap(self.builder, self.proxy_for(label, DType::Bool, shape))
    }
}

#[derive(Debug, Clone)]
struct Item {
    index: usize,
    contents: HashMap<String, NodeId>,
    scoped: HashMap<String, NodeId>,
}

/// One built item of a sequence.
#[derive(Debug, Clone)]
pub struct BuiltItem<'s>(&'s Item);

impl BuiltItem<'_> {
    pub fn index(&self) -> usize {
        self.0.index
    }

    /// The node added under `label` (unscoped, or the full scoped label).
    pub fn get(&self, label: &str) -> Option<NodeId> {
        self.0
            .contents
            .get(label)
            .or_else(|| self.0.scoped.get(label))
            .copied()
    }

    /// Every (unscoped label, node) in the item.
    pub fn contents(&self) -> impl Iterator<Item = (&str, NodeId)> + '_ {
        self.0.contents.iter().map(|(l, &id)| (l.as_str(), id))
    }
}

/// A built sequence of items.
#[derive(Debug, Clone)]
pub struct Sequence {
    identifier: usize,
    name: Option<String>,
    items: Vec<Item>,
}

impl Sequence {
    /// Numbers the sequences built with one graph builder: 1, 2, ...
    pub fn identifier(&self) -> usize {
        self.identifier
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn items(&self) -> Vec<BuiltItem<'_>> {
        self.items.iter().map(BuiltItem).collect()
    }

    pub fn item(&self, index: usize) -> Option<BuiltItem<'_>> {
        self.items.get(index).map(BuiltItem)
    }

    pub fn last_item(&self) -> Option<BuiltItem<'_>> {
        self.items.last().map(BuiltItem)
    }
}
