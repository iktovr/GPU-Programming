#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cassert>
#include <vector>
#include <cmath>
#include <mpi.h>

#include "../common/vec3.hpp"
#include "../common/error_checkers.hpp"

#define _i(i, j, k) (((i) + 1) * (block.y + 2) + ((j) + 1)) * (block.z + 2) + ((k) + 1)
#define _ib(i, j, k) ((i) * grid.y + (j)) * grid.z + (k)

int main(int argc, char *argv[]) {
	std::ios::sync_with_stdio(false);
	int numproc, proc_id;
	vec3i grid, block, id;
	vec3 l, start, end, h;
	double eps, u0;
	std::string out_filename;
	std::ofstream out_file;

#ifdef TIME
	double time_start, time_end;
#endif

	std::vector<double> data, next_data;
	std::vector<std::vector<double>> buffer(12);
	std::vector<double> row_buffer;

	MPI_Status status;
	MPI_Request request;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	if (proc_id == 0) {
		std::cin >> grid >> block >> out_filename >> eps >> l
		         >> start.z >> end.z >> start.x >> end.x >> start.y >> end.y
		         >> u0;

		MPI_Assert(grid.x * grid.y * grid.z == numproc);

		out_file.open(out_filename);
	};

	MPI_Bcast(&grid, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&start, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&end, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	id.x = proc_id / (grid.y * grid.z);
	id.y = proc_id % (grid.y * grid.z) / grid.z;
	id.z = proc_id % (grid.y * grid.z) % grid.z;

	h = l / (grid * block);

	data.resize((block.x + 2) * (block.y + 2) * (block.z + 2), u0);
	next_data.resize((block.x + 2) * (block.y + 2) * (block.z + 2));
	for (int i = 0; i < 4; ++i) {
		buffer[i].resize(block.y * block.z);
	}
	for (int i = 4; i < 8; ++i) {
		buffer[i].resize(block.x * block.z);
	}
	for (int i = 8; i < 12; ++i) {
		buffer[i].resize(block.x * block.y);
	}

#ifdef TIME
	if (proc_id == 0) {
		time_start = MPI_Wtime();
	}
#endif

	double eps_k, eps_temp;
	do {
		MPI_Barrier(MPI_COMM_WORLD);
		
		eps_k = 0;

		if (id.x + 1 < grid.x) { // right
			for (int i = 0; i < block.y; ++i) {
				for (int j = 0; j < block.z; ++j) {
					buffer[0][i * block.z + j] = data[_i(block.x-1, i, j)];
				}
			}
			MPI_Isend(buffer[0].data(), buffer[0].size(), MPI_DOUBLE, _ib(id.x+1, id.y, id.z), proc_id, MPI_COMM_WORLD, &request);
		}
		if (id.x > 0) { // left
			for (int i = 0; i < block.y; ++i) {
				for (int j = 0; j < block.z; ++j) {
					buffer[2][i * block.z + j] = data[_i(0, i, j)];
				}
			}
			MPI_Isend(buffer[2].data(), buffer[2].size(), MPI_DOUBLE, _ib(id.x-1, id.y, id.z), proc_id, MPI_COMM_WORLD, &request);
		}

		if (id.y + 1 < grid.y) { // back
			for (int i = 0; i < block.x; ++i) {
				for (int j = 0; j < block.z; ++j) {
					buffer[4][i * block.z + j] = data[_i(i, block.y-1, j)];
				}
			}
			MPI_Isend(buffer[4].data(), buffer[4].size(), MPI_DOUBLE, _ib(id.x, id.y+1, id.z), proc_id, MPI_COMM_WORLD, &request);
		}
		if (id.y > 0) { // front
			for (int i = 0; i < block.x; ++i) {
				for (int j = 0; j < block.z; ++j) {
					buffer[6][i * block.z + j] = data[_i(i, 0, j)];
				}
			}
			MPI_Isend(buffer[6].data(), buffer[6].size(), MPI_DOUBLE, _ib(id.x, id.y-1, id.z), proc_id, MPI_COMM_WORLD, &request);
		}

		if (id.z + 1 < grid.z) { // top
			for (int i = 0; i < block.x; ++i) {
				for (int j = 0; j < block.y; ++j) {
					buffer[8][i * block.y + j] = data[_i(i, j, block.z-1)];
				}
			}
			MPI_Isend(buffer[8].data(), buffer[8].size(), MPI_DOUBLE, _ib(id.x, id.y, id.z+1), proc_id, MPI_COMM_WORLD, &request);
		}
		if (id.z > 0) { // bottom
			for (int i = 0; i < block.x; ++i) {
				for (int j = 0; j < block.y; ++j) {
					buffer[10][i * block.y + j] = data[_i(i, j, 0)];
				}
			}
			MPI_Isend(buffer[10].data(), buffer[10].size(), MPI_DOUBLE, _ib(id.x, id.y, id.z-1), proc_id, MPI_COMM_WORLD, &request);
		}

		if (id.x + 1 < grid.x) {
			MPI_Irecv(buffer[1].data(), buffer[1].size(), MPI_DOUBLE, _ib(id.x+1, id.y, id.z), _ib(id.x+1, id.y, id.z), MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
		if (id.x > 0) {
			MPI_Irecv(buffer[3].data(), buffer[3].size(), MPI_DOUBLE, _ib(id.x-1, id.y, id.z), _ib(id.x-1, id.y, id.z), MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
		for (int i = 0; i < block.y; ++i) {
			for (int j = 0; j < block.z; ++j) {
				// right
				if (id.x + 1 < grid.x) {
					data[_i(block.x, i, j)] = buffer[1][i * block.z + j];
				} else {
					data[_i(block.x, i, j)] = end.x;
				}

				// left
				if (id.x > 0) {
					data[_i(-1, i, j)] = buffer[3][i * block.z + j];
				} else {
					data[_i(-1, i, j)] = start.x;
				}
			}
		}

		if (id.y + 1 < grid.y) {
			MPI_Irecv(buffer[5].data(), buffer[5].size(), MPI_DOUBLE, _ib(id.x, id.y+1, id.z), _ib(id.x, id.y+1, id.z), MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
		if (id.y > 0) {
			MPI_Irecv(buffer[7].data(), buffer[7].size(), MPI_DOUBLE, _ib(id.x, id.y-1, id.z), _ib(id.x, id.y-1, id.z), MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
		for (int i = 0; i < block.x; ++i) {
			for (int j = 0; j < block.z; ++j) {
				// back
				if (id.y + 1 < grid.y) {
					data[_i(i, block.y, j)] = buffer[5][i * block.z + j];
				} else {
					data[_i(i, block.y, j)] = end.y;
				}

				// front
				if (id.y > 0) {
					data[_i(i, -1, j)] = buffer[7][i * block.z + j];
				} else {
					data[_i(i, -1, j)] = start.y;
				}
			}
		}

		if (id.z + 1 < grid.z) {
			MPI_Irecv(buffer[9].data(), buffer[9].size(), MPI_DOUBLE, _ib(id.x, id.y, id.z+1), _ib(id.x, id.y, id.z+1), MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
		if (id.z > 0) {
			MPI_Irecv(buffer[11].data(), buffer[11].size(), MPI_DOUBLE, _ib(id.x, id.y, id.z-1), _ib(id.x, id.y, id.z-1), MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
		for (int i = 0; i < block.x; ++i) {
			for (int j = 0; j < block.y; ++j) {
				// top
				if (id.z + 1 < grid.z) {
					data[_i(i, j, block.z)] = buffer[9][i * block.y + j];
				} else {
					data[_i(i, j, block.z)] = end.z;
				}

				// bottom
				if (id.z > 0) {
					 data[_i(i, j, -1)] = buffer[11][i * block.y + j];
				} else {
					 data[_i(i, j, -1)] = start.z;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < block.x; ++i) {
			for (int j = 0; j < block.y; ++j) {
				for (int k = 0; k < block.z; ++k) {
					next_data[_i(i, j, k)] = 
						((data[_i(i-1, j, k)] + data[_i(i+1, j, k)]) / (h.x * h.x) + 
						 (data[_i(i, j-1, k)] + data[_i(i, j+1, k)]) / (h.y * h.y) + 
						 (data[_i(i, j, k-1)] + data[_i(i, j, k+1)]) / (h.z * h.z)) / 2 /
						(1 / (h.x * h.x) + 1 / (h.y * h.y) + 1 / (h.z * h.z));

					eps_k = std::max(eps_k, std::abs(next_data[_i(i, j, k)] - data[_i(i, j, k)]));
				}
			}
		}

		std::swap(data, next_data);

		MPI_Allreduce(&eps_k, &eps_temp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		eps_k = eps_temp;
	} while (eps_k > eps);

#ifdef TIME
	if (proc_id == 0) {
		time_end = MPI_Wtime();
		std::cout << (time_end - time_start) * 1000 << std::endl;
	}
#else

#ifndef PLOT
	row_buffer.resize(block.x);

	if (proc_id != 0) {
		for (int k = 0; k < block.z; ++k) {
			for (int j = 0; j < block.y; ++j) {
				for (int i = 0; i < block.x; ++i) {
					row_buffer[i] = data[_i(i, j, k)];
				}
				MPI_Send(row_buffer.data(), block.x, MPI_DOUBLE, 0, proc_id, MPI_COMM_WORLD);
			}
		}
	} else {
		out_file << std::setprecision(6) << std::scientific;
		for (int kb = 0; kb < grid.z; ++kb) {
			for (int k = 0; k < block.z; ++k) {
				for (int jb = 0; jb < grid.y; ++jb) {
					for (int j = 0; j < block.y; ++j) {
						for (int ib = 0; ib < grid.x; ++ib) {
							if (_ib(ib, jb, kb) == 0) {
								for (int i = 0; i < block.x; ++i) {
									row_buffer[i] = data[_i(i, j, k)];
								}
							} else {
								MPI_Recv(row_buffer.data(), block.x, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							}

							for (int i = 0; i < block.x; ++i) {
								out_file << row_buffer[i] << ' ';
							}
						}
						out_file << '\n';
					}
				}
				out_file << '\n';
			}
		}
	}

// Вторая версия вывода, для отрисовки тепловых карт через pgfplots
#else
	row_buffer.resize(block.z);

	if (proc_id != 0) {
		for (int i = 0; i < block.x; ++i) {
			for (int j = 0; j < block.y; ++j) {
				for (int k = 0; k < block.z; ++k) {
					row_buffer[k] = data[_i(i, j, k)];
				}
				MPI_Send(row_buffer.data(), block.z, MPI_DOUBLE, 0, proc_id, MPI_COMM_WORLD);
			}
		}
	} else {
		out_file << std::setprecision(6) << std::scientific;
		for (int ib = 0; ib < grid.x; ++ib) {
			for (int i = 0; i < block.x; ++i) {
				for (int jb = 0; jb < grid.y; ++jb) {
					for (int j = 0; j < block.y; ++j) {
						out_file << ib*block.x+i << ' ' << jb*block.y+j << ' ';
						for (int kb = 0; kb < grid.z; ++kb) {
							if (_ib(ib, jb, kb) == 0) {
								for (int k = 0; k < block.z; ++k) {
									row_buffer[k] = data[_i(i, j, k)];
								}
							} else {
								MPI_Recv(row_buffer.data(), block.z, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							}

							for (int k = 0; k < block.z; ++k) {
								out_file << row_buffer[k] << ' ';
							}
						}
						out_file << '\n';
					}
				}
			}
		}
	}
#endif
#endif

	MPI_Finalize();
	return 0;
}