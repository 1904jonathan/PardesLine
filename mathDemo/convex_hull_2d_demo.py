"""
2D Convex Hull - Graham Scan Algorithm Demonstration

Demonstrates convex hull computation for 5 points using:
- Polar angle sorting with atan2
- Cross product for turn detection
- Stack-based hull construction
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def find_anchor_point(points: np.ndarray) -> int:
    """Find anchor point (lowest y, leftmost if tie)."""
    min_y_idx = np.argmin(points[:, 1])
    min_y = points[min_y_idx, 1]
    candidates = np.where(points[:, 1] == min_y)[0]

    if len(candidates) > 1:
        return candidates[np.argmin(points[candidates, 0])]
    return min_y_idx


def compute_polar_angle(anchor: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute polar angles: θ = atan2(Δy, Δx)."""
    vectors = points - anchor
    return np.arctan2(vectors[:, 1], vectors[:, 0])


def sort_by_polar_angle(points: np.ndarray, anchor_idx: int) -> np.ndarray:
    """Sort points by polar angle from anchor."""
    anchor = points[anchor_idx]
    angles = compute_polar_angle(anchor, points)
    distances = np.linalg.norm(points - anchor, axis=1)

    indices = list(range(len(points)))
    indices.remove(anchor_idx)
    sorted_indices = sorted(indices, key=lambda i: (angles[i], distances[i]))

    return np.array([anchor_idx] + sorted_indices)


def cross_product_2d(O: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute 2D cross product to determine turn direction.

    cross = (A - O) × (B - O)

    Returns:
        > 0: CCW turn (left)
        < 0: CW turn (right)
        = 0: Collinear
    """
    OA = A - O
    OB = B - O
    return OA[0] * OB[1] - OA[1] * OB[0]


def graham_scan(points: np.ndarray) -> List[int]:
    """
    Graham Scan algorithm to compute convex hull.

    Algorithm:
    1. Find anchor point (lowest y)
    2. Sort points by polar angle
    3. Build hull using stack and cross product

    Returns:
        List of indices forming convex hull (CCW order)
    """
    n = len(points)
    if n < 3:
        return list(range(n))

    # Find anchor and sort
    anchor_idx = find_anchor_point(points)
    sorted_indices = sort_by_polar_angle(points, anchor_idx)
    sorted_points = points[sorted_indices]

    # Initialize stack with first 3 points
    stack = [0, 1, 2]

    # Process remaining points
    for i in range(3, n):
        while len(stack) > 1:
            O = sorted_points[stack[-2]]
            A = sorted_points[stack[-1]]
            B = sorted_points[i]

            cross = cross_product_2d(O, A, B)

            if cross > 0:
                break  # CCW turn, keep point
            else:
                stack.pop()  # CW or collinear, remove

        stack.append(i)

    # Convert to original indices
    return [sorted_indices[i] for i in stack]


def visualize_initial_points(points: np.ndarray):
    """Visualize initial point cloud before hull computation."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=150,
               alpha=0.7, label='Input points', zorder=3, edgecolors='darkblue', linewidths=2)

    # Point labels
    for i, point in enumerate(points):
        ax.annotate(f'P{i}\n({point[0]:.1f}, {point[1]:.1f})', (point[0], point[1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Formatting
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y coordinate', fontsize=12, fontweight='bold')
    ax.set_title('Initial Point Cloud (5 Points)\nBefore Convex Hull Computation',
                fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def visualize_convex_hull(points: np.ndarray, hull_indices: List[int]):
    """Visualize point cloud and convex hull."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=100,
               alpha=0.6, label='Input points', zorder=3)

    # Point labels
    for i, point in enumerate(points):
        ax.annotate(f'P{i}', (point[0], point[1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Hull vertices
    hull_points = points[hull_indices]
    ax.scatter(hull_points[:, 0], hull_points[:, 1],
              c='red', s=200, marker='o', alpha=0.8,
              label='Hull vertices', zorder=4, edgecolors='darkred', linewidths=2)

    # Hull edges
    hull_closed = np.vstack([hull_points, hull_points[0]])
    ax.plot(hull_closed[:, 0], hull_closed[:, 1],
           'r-', linewidth=2, alpha=0.7, label='Convex hull', zorder=2)

    # Anchor point
    anchor_idx = hull_indices[0]
    ax.scatter(points[anchor_idx, 0], points[anchor_idx, 1],
              c='green', s=400, marker='*',
              label='Anchor point', zorder=5, edgecolors='darkgreen', linewidths=2)

    # Formatting
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y coordinate', fontsize=12, fontweight='bold')
    ax.set_title('2D Convex Hull - Graham Scan Algorithm',
                fontsize=14, fontweight='bold', pad=20)

    # Algorithm explanation
    explanation = (
        "Graham Scan Steps:\n"
        "1. Anchor: lowest y point\n"
        "2. Sort: θ = atan2(Δy, Δx)\n"
        "3. Cross product:\n"
        "   cross > 0 → CCW (keep)\n"
        "   cross < 0 → CW (remove)\n"
        "4. Build hull with stack"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', bbox=props, family='monospace')

    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration with 5 points."""
    print("2D Convex Hull - Graham Scan Algorithm")
    print("="*50)

    # Define 5 points
    points = np.array([
        [2.0, 3.0],   # P0 - Interior point
        [1.0, 1.0],   # P1 - Bottom left (anchor)
        [4.0, 1.5],   # P2 - Bottom right
        [3.5, 4.0],   # P3 - Top right
        [1.5, 4.5],   # P4 - Top left
    ])

    print("\nInput points:")
    for i, p in enumerate(points):
        print(f"  P{i} = ({p[0]:.3f}, {p[1]:.3f})")

    # Visualize initial points
    print("\nVisualizing initial points...")
    visualize_initial_points(points)

    # Compute convex hull
    print("\nComputing convex hull...")
    hull_indices = graham_scan(points)

    print(f"\nConvex hull vertices (CCW order):")
    for i, idx in enumerate(hull_indices):
        p = points[idx]
        print(f"  {i+1}. P{idx} = ({p[0]:.3f}, {p[1]:.3f})")

    print(f"\nHull vertices: {len(hull_indices)} / {len(points)} points")

    # Interior points
    interior = set(range(len(points))) - set(hull_indices)
    if interior:
        print(f"Interior points: {sorted(interior)}")

    # Visualize final result
    print("\nVisualizing convex hull result...")
    visualize_convex_hull(points, hull_indices)


if __name__ == "__main__":
    main()
