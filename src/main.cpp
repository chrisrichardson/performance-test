// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "Elasticity.h"
#include "Poisson.h"
#include "elasticity_problem.h"
#include "mesh.h"
#include "poisson_problem.h"
#include <boost/program_options.hpp>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/subsystem.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/common/version.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <string>
#include <thread>
#include <utility>
#include "mem.h"

namespace po = boost::program_options;

void solve(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "problem_type", po::value<std::string>()->default_value("poisson"),
      "problem (poisson or elasticity)")(
      "mesh_type", po::value<std::string>()->default_value("cube"),
      "mesh (cube or unstructured)")(
      "scaling_type", po::value<std::string>()->default_value("weak"),
      "scaling (weak or strong)")(
      "output", po::value<std::string>()->default_value(""),
      "output directory (no output unless this is set)")(
      "ndofs", po::value<std::size_t>()->default_value(50000),
      "number of degrees of freedom");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return;
  }

  const std::string problem_type = vm["problem_type"].as<std::string>();
  const std::string mesh_type = vm["mesh_type"].as<std::string>();
  const std::string scaling_type = vm["scaling_type"].as<std::string>();
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  const std::string output_dir = vm["output"].as<std::string>();
  const bool output = (output_dir.size() > 0);

  bool strong_scaling;
  if (scaling_type == "strong")
    strong_scaling = true;
  else if (scaling_type == "weak")
    strong_scaling = false;
  else
    throw std::runtime_error("Scaling type '" + scaling_type + "` unknown");

  // Get number of processes
  const std::size_t num_processes = dolfinx::MPI::size(MPI_COMM_WORLD);

  // Assemble problem
  std::shared_ptr<dolfinx::mesh::Mesh> mesh;
  if (problem_type == "poisson")
  {
    dolfinx::common::Timer t0("ZZZ Create Mesh");
    auto cmap
        = dolfinx::fem::create_coordinate_map(create_coordinate_map_Poisson);
    if (mesh_type == "cube")
      mesh = create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1, cmap);
    else
      mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1, cmap);
    t0.stop();
  }
  else
    throw std::runtime_error("Unknown problem type: " + problem_type);

  // Print simulation summary
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    char petsc_version[256];
    PetscGetVersion(petsc_version, 256);

    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
    std::cout << "Test problem summary" << std::endl;
    std::cout << "  dolfinx version: " << DOLFINX_VERSION_STRING << std::endl;
    std::cout << "  dolfinx hash:    " << DOLFINX_VERSION_GIT << std::endl;
    std::cout << "  ufl hash:        " << UFC_SIGNATURE << std::endl;
    std::cout << "  petsc version:   " << petsc_version << std::endl;
    std::cout << "  Problem type:    " << problem_type << std::endl;
    std::cout << "  Scaling type:    " << scaling_type << std::endl;
    std::cout << "  Num processes:   " << num_processes << std::endl;
    std::cout << "  Mesh cells:   "
              << mesh->topology().index_map(3)->size_global() << std::endl;
    std::cout << "  Mesh vertices:   "
              << mesh->topology().index_map(0)->size_global() << std::endl;
    std::cout << "  Average vertices per process: "
              << mesh->topology().index_map(0)->size_global()
                     / dolfinx::MPI::size(MPI_COMM_WORLD)
              << std::endl;
    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
  }

  dolfinx::common::Timer t6("ZZZ Output");
  if (false)
  {
    std::string filename = "./mesh-" + std::to_string(num_processes) + ".xdmf";
    dolfinx::io::XDMFFile file(MPI_COMM_WORLD, filename, "w");
    file.write_mesh(*mesh);
  }
  t6.stop();

  // Display timings
  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
}

int main(int argc, char* argv[])
{
  dolfinx::common::subsystem::init_mpi();
  dolfinx::common::subsystem::init_logging(argc, argv);
  dolfinx::common::subsystem::init_petsc(argc, argv);
  std::string thread_name = "RANK: " 
    + std::to_string(dolfinx::MPI::rank(MPI_COMM_WORLD));
  loguru::set_thread_name(thread_name.c_str());
  loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

  const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  if (rank == 0)
  { 
    bool quit_flag = false;
    std::thread mem_thread(process_mem_usage, std::ref(quit_flag));
    solve(argc, argv);
    quit_flag = true;
    mem_thread.join();
  }
  else
    solve(argc, argv);

  dolfinx::common::subsystem::finalize_petsc();
  dolfinx::common::subsystem::finalize_mpi();
  return 0;
}
