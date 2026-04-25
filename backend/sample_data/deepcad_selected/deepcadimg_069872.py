import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.125, 0.2890625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.2571546052631579, 0.0).threePointArc((0.12857730263157896, 0.464618670865474), (0.0, 0.0)).close()
solid0=wp_sketch0.add(loop0).extrude(0.0625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.125, 0.2890625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.09769736842105264, -0.16825657894736842).lineTo(0.16011513157894738, -0.16825657894736842).lineTo(0.2578125, 0.0).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.0625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.03125, -0.125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.06513157894736843, 0.0).lineTo(0.06513157894736843, 0.2578125).lineTo(0.0, 0.2578125).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.0625)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(-0.2109375, 0.5078125, 0.0625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch3.moveTo(0.2131578947368421, 0.0).circle(0.2087171052631579)
solid3=wp_sketch3.add(loop3).extrude(-0.0546875)
solid=solid.cut(solid3)
# Generating a workplane for sketch 4
wp_sketch4 = cq.Workplane(cq.Plane(cq.Vector(-0.09375, 0.28125, 0.0625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop4=wp_sketch4.moveTo(0.07606907894736842, -0.13363486842105263).lineTo(0.11924342105263157, -0.13363486842105263).lineTo(0.1953125, 0.0).lineTo(0.0, 0.0).close()
solid4=wp_sketch4.add(loop4).extrude(-0.0546875)
solid=solid.cut(solid4)