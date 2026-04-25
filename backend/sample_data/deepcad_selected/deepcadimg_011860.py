import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.03125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.03125, 0.0).lineTo(0.03125, 0.03125).lineTo(0.0, 0.03125).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.15625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0, 0.3125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.03125, 0.0).lineTo(0.03125, 0.03125).lineTo(0.0, 0.03125).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.15625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.71875, -0.03125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.03125, 0.0).lineTo(0.03125, 0.03125).lineTo(0.0, 0.03125).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.15625)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(0.71875, 0.3125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch3.moveTo(0.03125, 0.0).lineTo(0.03125, 0.03125).lineTo(0.0, 0.03125).lineTo(0.0, 0.0).close()
solid3=wp_sketch3.add(loop3).extrude(0.15625)
solid=solid.union(solid3)