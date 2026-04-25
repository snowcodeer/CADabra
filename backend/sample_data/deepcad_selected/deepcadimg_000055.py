import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5, -0.25, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.0, 0.0).threePointArc((1.25, 0.25), (1.0, 0.5)).lineTo(0.0, 0.5).threePointArc((-0.25, 0.25), (0.0, 0.0)).close()
loop1=wp_sketch0.moveTo(0.0, 0.25).circle(0.13157894736842105)
loop2=wp_sketch0.moveTo(1.0, 0.25).circle(0.13157894736842105)
solid0=wp_sketch0.add(loop0).add(loop1).add(loop2).extrude(0.25)
solid=solid0