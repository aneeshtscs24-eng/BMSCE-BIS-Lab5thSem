import numpy as np

image=np.array([
    [10,12,15,200,210],
    [11,14,18,205,215],
    [9,13,16,195,220],
    [180,190,200,230,240],
    [175,185,195,225,235]
],dtype=np.uint8)

print("Original Matrix:")
print(image)

hist=np.bincount(image.flatten(),minlength=256)

def otsu_fitness(threshold,hist):
    threshold=int(round(threshold))
    threshold=np.clip(threshold,1,254)
    total=np.sum(hist)
    class0=np.sum(hist[:threshold+1])
    class1=np.sum(hist[threshold+1:])
    if class0==0 or class1==0:
        return 0
    w0=class0/total
    w1=class1/total
    i0=np.arange(0,threshold+1)
    i1=np.arange(threshold+1,256)
    mean0=np.sum(i0*hist[:threshold+1])/class0
    mean1=np.sum(i1*hist[threshold+1:])/class1
    return w0*w1*(mean0-mean1)**2

num_particles=5
num_iterations=10
w=0.7
c1=1.5
c2=1.5
lower_bound=1
upper_bound=254

np.random.seed(42)

positions=np.random.uniform(lower_bound,upper_bound,num_particles)
velocities=np.random.uniform(-10,10,num_particles)

fitness=np.array([
    otsu_fitness(p,hist) for p in positions
])

pbest_positions=positions.copy()
pbest_fitness=fitness.copy()

best_index=np.argmax(pbest_fitness)
gbest_position=pbest_positions[best_index]
gbest_fitness=pbest_fitness[best_index]

for iteration in range(num_iterations):
    r1=np.random.random(num_particles)
    r2=np.random.random(num_particles)

    velocities=(
        w*velocities+
        c1*r1*(pbest_positions-positions)+
        c2*r2*(gbest_position-positions)
    )

    positions+=velocities
    positions=np.clip(positions,lower_bound,upper_bound)

    fitness=np.array([
        otsu_fitness(p,hist) for p in positions
    ])

    improved=fitness>pbest_fitness
    pbest_positions[improved]=positions[improved]
    pbest_fitness[improved]=fitness[improved]

    best_index=np.argmax(pbest_fitness)

    if pbest_fitness[best_index]>gbest_fitness:
        gbest_fitness=pbest_fitness[best_index]
        gbest_position=pbest_positions[best_index]

    print("Iteration",iteration+1,"Threshold:",int(round(gbest_position)))

threshold=int(round(gbest_position))

segmented=np.where(image<=threshold,0,255)

print("\nOptimal Threshold:",threshold)
print("\nSegmented Matrix:")
print(segmented)