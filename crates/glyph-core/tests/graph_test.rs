use glyph_core::{Contour, ContourInstance, GraphConfig, NODE_DIM, build_graph};

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
        ContourInstance {
            contour: square(0.0, 0.0, 300.0),
            char_id: 0,
        },
        ContourInstance {
            contour: square(150.0, 0.0, 300.0),
            char_id: 1,
        },
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
        ContourInstance {
            contour: square(0.0, 0.0, 200.0),
            char_id: 0,
        },
        // 5 em away: far beyond the default 0.25em connection radius.
        ContourInstance {
            contour: square(5000.0, 0.0, 200.0),
            char_id: 1,
        },
    ];
    let g = build_graph(&contours, upem, &GraphConfig::default());
    // No cross-character edges should exist at this distance.
    assert!(g.edge_labels.iter().all(|&l| l == 1));
}

#[test]
fn contour_bridge_links_distant_strokes_of_one_glyph() {
    let upem = 1000.0;
    // Two components of one glyph separated enough that kNN does not link them;
    // a neighbouring glyph sits within bridge range of the right-hand stroke.
    let contours = vec![
        ContourInstance {
            contour: square(0.0, 0.0, 50.0),
            char_id: 0,
        },
        ContourInstance {
            contour: square(360.0, 0.0, 50.0),
            char_id: 0,
        },
        ContourInstance {
            contour: square(450.0, 0.0, 50.0),
            char_id: 1,
        },
    ];
    let no_bridge = GraphConfig {
        contour_bridge: 0.0,
        ..GraphConfig::default()
    };
    let with_bridge = GraphConfig {
        contour_bridge: 0.35,
        ..GraphConfig::default()
    };
    let g0 = build_graph(&contours, upem, &no_bridge);
    let g1 = build_graph(&contours, upem, &with_bridge);

    let pos0 = g0.edge_labels.iter().filter(|&&l| l == 1).count();
    let pos1 = g1.edge_labels.iter().filter(|&&l| l == 1).count();
    let neg1 = g1.edge_labels.iter().filter(|&&l| l == 0).count();

    assert!(pos0 < pos1, "bridges should add same-glyph positive edges");
    assert!(neg1 > 0, "bridges should add cross-glyph negative edges");
    assert!(g1.num_edges() > g0.num_edges());
}
