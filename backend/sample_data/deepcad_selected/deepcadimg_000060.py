import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.421875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.296875, 0.0).lineTo(1.296875, 1.0648026315789474).lineTo(0.0, 1.0648026315789474).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.24572368421052632, 0.08190789473684211).circle(0.05460526315789474)
loop2=wp_sketch0.moveTo(0.24572368421052632, 0.9692434210526316).circle(0.05460526315789474)
loop3=wp_sketch0.moveTo(0.6962171052631579, 0.5324013157894737).circle(0.32763157894736844)
loop4=wp_sketch0.moveTo(1.1330592105263158, 0.08190789473684211).circle(0.05460526315789474)
loop5=wp_sketch0.moveTo(1.1330592105263158, 0.9692434210526316).circle(0.05460526315789474)
solid0=wp_sketch0.add(loop0).add(loop1).add(loop2).add(loop3).add(loop4).add(loop5).extrude(0.140625)
solid=solid0