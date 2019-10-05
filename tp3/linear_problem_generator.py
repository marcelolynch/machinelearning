from random import random
import matplotlib.pyplot as plt
import csv

MIN_X = 0
MAX_X = 5
MIN_Y = 0
MAX_Y = 5


def produce(n, m = 1, b = 0, prob = 0.5, delta = 0):
    data = []
    while n > 0:
        klass = -1 if random() < prob else 1
        x = MIN_X + (MAX_X - MIN_Y) * random()

        # Dado x, busco los limites de y para cada clase (region)
        if klass == -1:   # Abajo de la recta
            low_y = MIN_Y
            high_y = min(MAX_Y, m*x + b - delta) 
        else:   # Arriba de la recta
            low_y = max(m*x + b + delta, 0)
            high_y = MAX_Y

        if low_y >= high_y: # Podria pasar
            continue        # Probar de nuevo, total tenemos tiempo

        y = low_y + random()*(high_y - low_y)
        n -= 1
        data.append((x,y,klass))
    return data

def save(filename, data):
    with open(filename, 'w', newline='') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['x','y','class'])
        for d in data:
            csv_out.writerow(d)

def plot(data, mb = None) :
    plt.style.use('ggplot')
    plt.scatter([x[0] for x in data if x[2] == 1], [x[1] for x in data if x[2] == 1], marker='.', c='b')
    plt.scatter([x[0] for x in data if x[2] == -1], [x[1] for x in data if x[2] == -1], marker='.', c='r')
    if mb is not None:
        m = mb[0]
        b = mb[1]
        plt.plot([i for i in range(6)], [m*i + b for i in range(6)], 'k')
    plt.xlim((0,5))
    plt.ylim((0,5))
    plt.show()


if __name__ == "__main__":
    m = 0
    b = 2.5
    data = produce(400, m, b, prob=0.5, delta = 0.3)
    save('data/linsep.csv', data)
    plot(data, (m,b))
