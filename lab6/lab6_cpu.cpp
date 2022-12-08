#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <random>

#include <GL/glew.h>
#include <GL/freeglut.h>

#ifdef TIME
#include <iostream>
#include <chrono>
using namespace std::chrono;
#endif

#include "../common/vec3.hpp"

using std::min;
using std::max;
using std::abs;
using std::sqrt;

using uchar = unsigned char;

struct point_charge {
	vec3 pos;
	vec3 v;
	double q;
};

template <class T>
inline T cube(const T& x) {
	return x * x * x;
}

template <class T>
inline T sqr(const T& x) {
	return x * x;
}

const double PI = 2 * std::acos(0);

int width = 1024, height = 648;

const double max_speed = 10;
const double acceleration = 0.5;

const double box = 15.0;
const int field_size = 100;
const double eps = 1e-3;
const double K = 100.0;
const double gravity = 0.06;
const double speed_decay = 0.999;
const double camera_speed_decay = 0.99;
const double shift_z = 0.75;
const double dt = 0.005;

std::vector<bool> keys(256, false);

GLUquadric* quadratic;
GLuint textures[3];

const double object_charge = 1;
const double object_radius = 0.8;
const double bullet_radius = 0.6;
const int object_count = 150;

std::vector<point_charge> objects;
point_charge camera{{-box, 0, box}, {0, 0, 0}, 5};
double yaw = 0, pitch = 0, dyaw = 0, dpitch = 0;
bool bullet_active = false;
point_charge bullet{{0, 0, 10000}, {0, 0, 0}, 10};
std::vector<uchar> field_data(field_size * field_size * 4, 0);

#ifdef TIME
high_resolution_clock::time_point prev_time_since_start, time_since_start;
bool first = true;
#endif

// Загрузка текстуры из изображения
void load_texture(const char file[], GLuint texture) {
	int wt, ht;
	std::ifstream tex_file(file, std::ios::binary);
	tex_file.read(reinterpret_cast<char*>(&wt), sizeof(int));
	tex_file.read(reinterpret_cast<char*>(&ht), sizeof(int));
	std::vector<uchar> data(wt * ht * 4);
	tex_file.read(reinterpret_cast<char*>(data.data()), 4 * wt * ht);

	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, /*GL_NEAREST*/ GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, /*GL_NEAREST*/ GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

// Генерация текстуры пола
void field(std::vector<uchar>& field_data) {
	double x, y, f;
	for (int i = 0; i < field_size; ++i) {
		for (int j = 0; j < field_size; ++j) {
			f = 0;
			x = (2.0 * i / (field_size - 1.0) - 1.0) * box;
			y = (2.0 * j / (field_size - 1.0) - 1.0) * box;
			for (auto& object: objects) {
				f += K * object.q / (sqr(x - object.pos.x) + sqr(y - object.pos.y) + sqr(shift_z - object.pos.z) + eps);
			}
			if (bullet_active) {
				f += K * bullet.q / (sqr(x - bullet.pos.x) + sqr(y - bullet.pos.y) + sqr(shift_z - bullet.pos.z) + eps);
			}
			f = min(f, 255.0);
			field_data[(j * field_size + i) * 4 + 1] = (uchar)f;
		}
	}
}

void update_objects(double dt) {
	for (size_t i = 0; i < objects.size(); ++i) {
		// Замедление
		objects[i].v *= speed_decay;

		// Гравитация
		objects[i].v.z -= gravity;

		// Отталкивание от стен
		objects[i].v.x += K * objects[i].q * objects[i].q * (objects[i].pos.x - box) / (cube(abs(objects[i].pos.x - box)) + eps) * dt;
		objects[i].v.x += K * objects[i].q * objects[i].q * (objects[i].pos.x + box) / (cube(abs(objects[i].pos.x + box)) + eps) * dt;

		objects[i].v.y += K * objects[i].q * objects[i].q * (objects[i].pos.y - box) / (cube(abs(objects[i].pos.y - box)) + eps) * dt;
		objects[i].v.y += K * objects[i].q * objects[i].q * (objects[i].pos.y + box) / (cube(abs(objects[i].pos.y + box)) + eps) * dt;

		objects[i].v.z += K * objects[i].q * objects[i].q * (objects[i].pos.z - 2 * box) / (cube(abs(objects[i].pos.z - 2 * box)) + eps) * dt;
		objects[i].v.z += K * objects[i].q * objects[i].q * (objects[i].pos.z) / (cube(abs(objects[i].pos.z)) + eps) * dt;

		// Отталкивание от камеры
		objects[i].v += K * camera.q * objects[i].q * (objects[i].pos - camera.pos) / (cube((objects[i].pos - camera.pos).length()) + eps) * dt;

		// Отталкивание от пули
		if (bullet_active) {
			objects[i].v += K * bullet.q * objects[i].q * (objects[i].pos - bullet.pos) / (cube((objects[i].pos - bullet.pos).length()) + eps) * dt;
		}

		// Отталкивание от других объектов
		for (size_t j = 0; j < objects.size(); ++j) {
			if (j == i) {
				continue;
			}
			objects[i].v += K * objects[j].q * objects[i].q * (objects[i].pos - objects[j].pos) / (cube((objects[i].pos - objects[j].pos).length()) + eps) * dt;
		}
	}

	for (size_t i = 0; i < objects.size(); ++i) {
		objects[i].pos += objects[i].v * dt;
	}
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(camera.pos.x, camera.pos.y, camera.pos.z,
	          camera.pos.x + cos(yaw) * cos(pitch),
	          camera.pos.y + sin(yaw) * cos(pitch),
	          camera.pos.z + sin(pitch),
	          0.0f, 0.0f, 1.0f);

	// Сферы
	glBindTexture(GL_TEXTURE_2D, textures[1]);
	static float angle = 0.0;

	for (auto& object: objects) {
		glPushMatrix();
			glTranslatef(object.pos.x, object.pos.y, object.pos.z);
			glRotatef(angle, 0.0, 0.0, 1.0);
			gluSphere(quadratic, object_radius, 16, 16);
		glPopMatrix();
	}
	angle += 0.15;

	// Пуля
	if (bullet_active) {
		glBindTexture(GL_TEXTURE_2D, textures[2]);
		glPushMatrix();
			glTranslatef(bullet.pos.x, bullet.pos.y, bullet.pos.z);
			gluSphere(quadratic, bullet_radius, 16, 16);
		glPopMatrix();
	}

	// Пол
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)field_size, (GLsizei)field_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, field_data.data()); 
	
	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-box, -box, 0.0);

		glTexCoord2f(1.0, 0.0);
		glVertex3f(box, -box, 0.0);

		glTexCoord2f(1.0, 1.0);
		glVertex3f(box, box, 0.0);

		glTexCoord2f(0.0, 1.0);
		glVertex3f(-box, box, 0.0);
	glEnd();

	
	glBindTexture(GL_TEXTURE_2D, 0);

	// Куб
	glLineWidth(2);
	glColor3f(0.5f, 0.5f, 0.5f);
	glBegin(GL_LINES);
		glVertex3f(-box, -box, 0.0);
		glVertex3f(-box, -box, 2.0 * box);

		glVertex3f(box, -box, 0.0);
		glVertex3f(box, -box, 2.0 * box);

		glVertex3f(box, box, 0.0);
		glVertex3f(box, box, 2.0 * box);

		glVertex3f(-box, box, 0.0);
		glVertex3f(-box, box, 2.0 * box);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(-box, -box, 0.0);
		glVertex3f(box, -box, 0.0);
		glVertex3f(box, box, 0.0);
		glVertex3f(-box, box, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(-box, -box, 2.0 * box);
		glVertex3f(box, -box, 2.0 * box);
		glVertex3f(box, box, 2.0 * box);
		glVertex3f(-box, box, 2.0 * box);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
}


// Обработка нажатых кнопок
void process_input() {
	if (keys['w']) {                 // "W" Движение вперед
		camera.v.x += cos(yaw) * cos(pitch) * acceleration;
		camera.v.y += sin(yaw) * cos(pitch) * acceleration;
		camera.v.z += sin(pitch) * acceleration;
	}

	if (keys['s']) {                 // "S" Назад
		camera.v.x += -cos(yaw) * cos(pitch) * acceleration;
		camera.v.y += -sin(yaw) * cos(pitch) * acceleration;
		camera.v.z += -sin(pitch) * acceleration;
	}

	if (keys['a']) {                 // "A" Влево
		camera.v.x += -sin(yaw) * acceleration;
		camera.v.y += cos(yaw) * acceleration;
	}

	if (keys['d']) {                 // "D" Вправо
		camera.v.x += sin(yaw) * acceleration;
		camera.v.y += -cos(yaw) * acceleration;
	}

	if (keys[' ']) {                 // "space" Вверх
		camera.v.z += acceleration;
	}

	if (keys['c']) {                 // "C" Вниз
		camera.v.z -= acceleration;
	}
}

void update() {
	process_input();

#ifdef TIME
	if (first) {
		first = false;
		prev_time_since_start = high_resolution_clock::now();
	} else {
		time_since_start = high_resolution_clock::now();
		std::cout << duration_cast<milliseconds>(time_since_start - prev_time_since_start).count() << '\n';
		prev_time_since_start = time_since_start;
	}
#endif

	// Ограничение максимальной скорости
	float v = camera.v.length();
	if (v > max_speed) {
		camera.v *= max_speed / v;
	}
	camera.pos += camera.v * dt;
	camera.v *= camera_speed_decay;
	// Пол, ниже которого камера не может переместиться
	if (camera.pos.z < 1.0) {
		camera.pos.z = 1.0;
		camera.v.z = 0.0;
	}
	// Вращение камеры
	if (abs(dpitch) + abs(dyaw) > 0.0001) {
		yaw += dyaw;
		pitch += dpitch;
		pitch = min(PI / 2.0f - 0.0001f, max(-PI / 2.0f + 0.0001f, pitch));
		dyaw = dpitch = 0.0;
	}

	if (bullet_active) {
		bullet.pos += bullet.v * dt;
		if (bullet.pos.length() > box * 10) {
			bullet_active = false;
		}
	}

	update_objects(dt);

	// field(field_data);

	glutPostRedisplay();
}

// Обработка ввода
void key_pressed(uchar key, int, int) {
	if (key == 27) {                 // "escape" Выход
		glDeleteTextures(2, textures);
		gluDeleteQuadric(quadratic);
		exit(0);
	} else if (key == 'x') {         // "x" полная остановка
		for (size_t i = 0; i < keys.size(); ++i) {
			keys[i] = false;
		}
		camera.v *= 0;
	} else {                         // остальные запоминаются
		keys[key] = true;
	}
}

void key_released(uchar key, int, int) {
	keys[key] = false;
}

void mouse(int x, int y) {
	static int x_prev = width / 2, y_prev = height / 2;
	float dx = 0.005 * (x - x_prev);
    float dy = 0.005 * (y - y_prev);
	dyaw -= dx;
    dpitch -= dy;
	x_prev = x;
	y_prev = y;

	// Перемещаем указатель мышки в центр, когда он достиг границы
	if ((x < 20) || (y < 20) || (x > width - 20) || (y > height - 20)) {
		glutWarpPointer(width / 2, height / 2);
		x_prev = width / 2;
		y_prev = height / 2;
    }
}

void mouse_button(int button, int state, int, int) {
	if (state == GLUT_UP && button == GLUT_LEFT_BUTTON) {
		bullet_active = true;
		bullet.pos = camera.pos;
		bullet.v = {cos(yaw) * cos(pitch), sin(yaw) * cos(pitch), sin(pitch)};
		bullet.v.set_length(max_speed);
		bullet.pos += bullet.v * 0.05;
	}
}

void reshape(int new_width, int new_height) {
	width = new_width;
	height = new_height;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
}

int main(int argc, char **argv) {
#ifdef TIME
	std::ios::sync_with_stdio(false);
#endif

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(key_pressed);
	glutKeyboardUpFunc(key_released);
	glutPassiveMotionFunc(mouse);
	glutMouseFunc(mouse_button);
	glutReshapeFunc(reshape);

	glutSetCursor(GLUT_CURSOR_NONE);

	glGenTextures(3, textures);
	load_texture("venus.data", textures[1]);
	load_texture("neptune.data", textures[2]);
	
	quadratic = gluNewQuadric();
	gluQuadricTexture(quadratic, GL_TRUE);

	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_TEXTURE_2D);
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glewInit();

	for (int i = 0; i < field_size; ++i) {
		for (int j = 0; j < field_size; ++j) {
			field_data[(j * field_size + i) * 4 + 3] = 255;
		}
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-box, box);
	for (int i = 0; i < object_count; ++i) {
		objects.push_back({{dis(gen), dis(gen), dis(gen) + box}, {0, 0, 0}, object_charge});
	}

	glutMainLoop();
}
