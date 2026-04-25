import cadquery as cq
w0=cq.Workplane('ZX',origin=(0,15,0))
r=w0.sketch().segment((-49,-7),(-34,-19)).arc((38,-36),(14,33)).segment((14,34)).segment((-7,50)).close().assemble().finalize().extrude(-30)