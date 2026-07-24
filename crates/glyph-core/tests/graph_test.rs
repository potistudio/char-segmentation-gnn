use glyph_core::{build_graph, Contour, ContourInstance, GraphConfig, NODE_DIM};

fn square(cx: f32, cy: f32, half: f32) -> Contour {
    Contour {
        points: vec![
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
    }
}

#[test]
fn builds_labeled_graph_from_two_characters() {
    let upem = 1000.0;
    // Two "characters": overlapping squares on the left, one square far right.
    let contours = vec![
        ContourInstance { contour: square(0.0, 0.0, 300.0), char_id: 0 },
        ContourInstance { contour: square(150.0, 0.0, 300.0), char_id: 1 },
    ];
    let g = build_graph(&contours, upem, &GraphConfig::default());

    assert!(g.num_nodes > 0);
    assert_eq!(g.node_dim as usize, NODE_DIM);
    assert_eq!(g.node_features.len(), g.num_nodes as usize * NODE_DIM);
    let e = g.num_edges();
    assert!(e > 0);
    assert_eq!(g.edge_index.len(), 2 * e);
    assert_eq!(g.edge_labels.len(), e);

    // Overlapping squares must produce both positive and negative edges.
    let pos = g.edge_labels.iter().filter(|&&l| l == 1).count();
    let neg = e - pos;
    assert!(pos > 0, "expected same-character edges");
    assert!(neg > 0, "expected cross-character edges");

    // Edges are stored symmetrically: same count of s->d and d->s pairs.
    let (src, dst) = g.edge_index.split_at(e);
    let mut fwd: Vec<(u32, u32)> = src.iter().copied().zip(dst.iter().copied()).collect();
    let mut rev: Vec<(u32, u32)> = dst.iter().copied().zip(src.iter().copied()).collect();
    fwd.sort_unstable();
    rev.sort_unstable();
    assert_eq!(fwd, rev);
}

#[test]
fn distant_contours_are_disconnected() {
    let upem = 1000.0;
    let contours = vec![
        ContourInstance { contour: square(0.0, 0.0, 200.0), char_id: 0 },
        // 5 em away: far beyond the default 0.25em connection radius.
        ContourInstance { contour: square(5000.0, 0.0, 200.0), char_id: 1 },
    ];
    let g = build_graph(&contours, upem, &GraphConfig::default());
    // No cross-character edges should exist at this distance.
    assert!(g.edge_labels.iter().all(|&l| l == 1));
}
