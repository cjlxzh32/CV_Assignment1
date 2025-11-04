#!/usr/bin/env python3
"""
Correctly analyze dense point cloud density by handling coordinate units
"""

from plyfile import PlyData
import numpy as np

def analyze_dense_ply(scene):
    """Analyze dense point cloud with correct unit handling"""
    
    ply_path = f'project/3D_Reconstruction_Pipeline/result/{scene}/dense_model/dense.ply'
    
    try:
        ply = PlyData.read(ply_path)
        vertex_data = ply['vertex']
        
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        points = np.column_stack([x, y, z])
        
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        bbox_size = max_pt - min_pt
        max_dim = bbox_size.max()
        
        # Determine unit system
        if max_dim > 1000:
            unit = "millimeters"
            scale = 1000.0
        elif max_dim > 100:
            unit = "centimeters"
            scale = 100.0
        else:
            unit = "meters"
            scale = 1.0
        
        # Calculate density in meters³
        volume_m3 = np.prod(bbox_size / scale)
        density_per_m3 = len(points) / volume_m3
        
        print(f"\n{scene}:")
        print(f"  Points: {len(points):,}")
        print(f"  Bounding box (original units): [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]")
        print(f"  Coordinate system: {unit}")
        print(f"  Bounding box (meters): [{bbox_size[0]/scale:.2f}, {bbox_size[1]/scale:.2f}, {bbox_size[2]/scale:.2f}]")
        print(f"  Volume: {volume_m3:.2f} m³")
        print(f"  Density: {density_per_m3:.2f} points/m³")
        
        # Assessment
        if 1 <= density_per_m3 <= 50:
            print(f"  ✅ Density meets standard (1-50 points/m³)")
        elif density_per_m3 > 50:
            print(f"  ✅ Density EXCEEDS standard (very dense!)")
        else:
            print(f"  ⚠️  Density below standard (<1 points/m³)")
        
        return {
            'points': len(points),
            'density': density_per_m3,
            'meets_standard': 1 <= density_per_m3 <= 50
        }
        
    except Exception as e:
        print(f"{scene}: Error - {e}")
        return None

def main():
    """Main analysis"""
    
    print("="*60)
    print("Correct Dense Point Cloud Density Analysis")
    print("="*60)
    print("\nAnalyzing with correct coordinate units...\n")
    
    scenes = ['arch', 'chinese_heritage_centre', 'pavilion']
    results = []
    
    for scene in scenes:
        result = analyze_dense_ply(scene)
        if result:
            results.append((scene, result))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    print(f"{'Scene':<25} {'Points':<12} {'Density (pts/m³)':<18} {'Status'}")
    print("-" * 70)
    
    all_meet_standard = True
    for scene, r in results:
        if r['density'] >= 1:
            status = "✅ Meets standard"
            if r['density'] > 50:
                status = "✅ EXCEEDS standard"
        else:
            status = "⚠️  Below standard"
            all_meet_standard = False
        print(f"{scene:<25} {r['points']:>11,} {r['density']:>17.2f} {status}")
    
    print(f"\n{'='*60}")
    if all_meet_standard:
        print("✅ All scenes meet density standard!")
        print("   No need to retake photos.")
    else:
        print("⚠️  Some scenes below standard")
        print("   Recommendations:")
        for scene, r in results:
            if not r['meets_standard']:
                print(f"   - {scene}: {r['points']:,} points, density {r['density']:.2f}")
                print(f"     Consider: More overlapping views, better texture")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

