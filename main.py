import plotly
import numpy as np
from stl import mesh
import plotly.graph_objects as go
import random


# Функция конвертации STL модели в Mesh3d
def stl2mesh3d(stl_mesh):
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K


# функция создания блочной сетки для отображения температуры
def create_temp_mesh3d(x_start, x_end, y_start, y_end, z_start, z_end, length, width, height, dx=25, dy=25, dz=25):
    l, w, h = (length, width, height)  # параметры одного блока
    x_n, y_n, z_n = int((x_end-x_start)/(l+dx)), int((y_end-y_start)/(w+dy)), int((z_end-z_start)/(h+dz))
    try:
        dx, dy, dz = (x_end-x_start-l*x_n)/(x_n-1), (y_end-y_start-w*y_n)/(y_n-1), (z_end-z_start-h*z_n)/(z_n-1)
    except ZeroDivisionError:
        raise Exception('Incorrect block parameters')

    # mesh параметры для одного блока
    x_ = np.array([x_start, x_start, x_start+l, x_start+l, x_start, x_start, x_start+l, x_start+l])
    y_ = np.array([y_start, y_start+w, y_start+w, y_start, y_start, y_start+w, y_start+w, y_start])
    z_ = np.array([z_start, z_start, z_start, z_start, z_start+h, z_start+h, z_start+h, z_start+h])

    i_ = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
    j_ = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
    k_ = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

    n = len(x_)
    nv = len(i_)
    x, y, z = np.zeros(n * x_n * y_n * z_n), np.zeros(n * x_n * y_n * z_n), np.zeros(n * x_n * y_n * z_n)
    I, J, K = np.zeros(nv * x_n * y_n * z_n), np.zeros(nv * x_n * y_n * z_n), np.zeros(nv * x_n * y_n * z_n)

    # копирование блока вдоль оси x
    for j in range(x_n):
        for i in range(n):
            x[i + j * n] = x_[i] + j * (l + dx)
            y[i + j * n] = y_[i]
            z[i + j * n] = z_[i]
        for i in range(nv):
            I[i + j * nv] = i_[i] + j * n
            J[i + j * nv] = j_[i] + j * n
            K[i + j * nv] = k_[i] + j * n
    n *= x_n
    nv *= x_n
    # копирование вдоль оси y
    for j in range(1, y_n):
        for i in range(n):
            x[i + j * n] = x[i]
            y[i + j * n] = y[i] + j * (w + dy)
            z[i + j * n] = z[i]
        for i in range(nv):
            I[i + j * nv] = I[i] + j * n
            J[i + j * nv] = J[i] + j * n
            K[i + j * nv] = K[i] + j * n
    n *= y_n
    nv *= y_n
    # копирование вдоль оси z
    for j in range(1, z_n):
        for i in range(n):
            x[i + j * n] = x[i]
            y[i + j * n] = y[i]
            z[i + j * n] = z[i] + j * (h + dz)
        for i in range(nv):
            I[i + j * nv] = I[i] + j * n
            J[i + j * nv] = J[i] + j * n
            K[i + j * nv] = K[i] + j * n
    return x, y, z, I, J, K


def set_temp_area(x, y, z):
    temp_array = np.zeros(len(x))
    for i in range(len(x)):
        temp_array[i] = set_temp(x, y, z)
    return temp_array


# определение случайной температуры по координатам
def set_temp(x, y, z):
    n = 10
    return sum([random.randint(25, 65) for _ in range(n)]) / n


def main():
    my_mesh = mesh.Mesh.from_file('sthe_stl_model_22606154.stl')
    vertices, I, J, K = stl2mesh3d(my_mesh)
    x, y, z = vertices.T
    colorscale = [[0, '#555555'], [1, '#e5dee5']]
    mesh3D = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        opacity=0.999999999999,
        hoverinfo='skip',
        flatshading=True,
        colorscale=colorscale,
        intensity=z,
        name='LOTUS STHE',
        showscale=False
    )

    x, y, z, I, J, K = create_temp_mesh3d(
        x_start=-2750, x_end=400,
        y_start=-160, y_end=160,
        z_start=-160, z_end=160,
        length=130, width=150, height=40,
        dx=40, dy=10, dz=15,
    )
    temp_area = set_temp_area(x, y, z)
    temp_mesh3d = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        hovertemplate='temp: %{intensity}',
        hoverlabel=dict(bgcolor='black', namelength=0),
        intensity=temp_area,
        opacity=0.999999999999,
        flatshading=True,
        name='TEMP',
        colorscale='Inferno',
    )

    title = "Mesh3d LOTUS STHE"
    layout = go.Layout(
        paper_bgcolor='white',
        title_text=title,
        title_x=0.5,
        font_color='black',
        width=1600,
        height=800,
        scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
        scene_xaxis_visible=True,
        scene_yaxis_visible=True,
        scene_zaxis_visible=True,
        scene=dict(aspectratio=dict(x=4, y=1, z=1)),
    )

    fig = go.Figure(data=[mesh3D, temp_mesh3d], layout=layout)
    # fig.show()
    plotly.offline.plot(fig, filename='file.html')


if __name__ == '__main__':
    main()
