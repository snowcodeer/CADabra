import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.25, 0.0), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.25263157894736843, 0.0).circle(0.25263157894736843)
solid0=wp_sketch0.add(loop0).extrude(0.375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.375, 0.1015625, -0.0859375), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop1=wp_sketch1.moveTo(0.05131578947368422, 0.0).circle(0.05131578947368422)
solid1=wp_sketch1.add(loop1).extrude(0.375)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.375, -0.203125, -0.0859375), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop2=wp_sketch2.moveTo(0.05131578947368422, 0.0).circle(0.05131578947368422)
solid2=wp_sketch2.add(loop2).extrude(0.375)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(0.375, 0.0, 0.125), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop3=wp_sketch3.moveTo(0.0, 0.05131578947368422).lineTo(0.05131578947368422, 0.05131578947368422).threePointArc((-0.03628574271878335, 0.08760153219246757), (0.0, 0.0)).close()
solid3=wp_sketch3.add(loop3).extrude(0.375)
solid=solid.union(solid3)
# Generating a workplane for sketch 4
wp_sketch4 = cq.Workplane(cq.Plane(cq.Vector(0.375, 0.0, 0.125), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop4=wp_sketch4.moveTo(0.0, 0.0).threePointArc((0.03314563036811942, 0.01372936963188057), (0.046875, 0.046875)).lineTo(0.0, 0.046875).lineTo(0.0, 0.0).close()
solid4=wp_sketch4.add(loop4).extrude(0.375)
solid=solid.union(solid4)