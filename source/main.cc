/**
 * @file main.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 *
 * @date 2022-01-14, Instationary Navier Stokes, JPT
 * @date 2021-10-13, Instationary Stokes, JR
 * @date 2019-11-05, DTM++/Stokes, UK
 *
 * @date 2018-12-18, Forked from DTM++.Project/dwr-heat as dwr-diffusion, UK,
 * MPB
 * @date 2018-07-26, DTM::Problem, UK
 * @date 2018-07-25, tested and running instationary version, UK, MPB
 * @date 2018-03-06, new implementation, UK
 * @date 2017-08-01, Heat/DWR, UK
 * @date 2016-01-15, condiff/SUPG, UK
 * @date 2016-01-12, UK
 * @date 2015-11-11, UK
 *
 * @brief Time-dependent Stokes solver
 *
 * @mainpage
 * The dwr-fluid subproject of the DTM++.Project covers to
 * simulate the time-dependent Navier-Stokes equation
 * \f[
 * \partial_t \boldsymbol b - \nabla \cdot (
 *   \nu\, 2\, \boldsymbol \epsilon(\boldsymbol b - \boldsymbol b_0)
 *   - (p - p_0)\, \boldsymbol 1) = \boldsymbol f\,,
 * \f]
 * for a slow-moving viscous incompressible fluid
 * \f[
 * - \nabla \cdot (\boldsymbol b - \boldsymbol b_0) = 0\,.
 * \f]
 * in the space-time cyclinder \f$ \Omega \times I \f$,
 * \f$ \Omega \subset \mathbb{R}^d \f$, \f$ I = (0,T) \f$ with \f$ T < \infty
 * \f$, equipped with appropriate inital and boundary values. with the partition
 * of the boundary \f$ \partial \Omega = \Gamma_D \cup \Gamma_N \f$ and \f$
 * \Gamma_D \neq \emptyset \f$.
 *
 *
 * Remark. DTM++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the License,
 * or (at your option) any later version.
 *
 * If you write scientific publication using results obtained by reusing parts
 * of DTM++, or specifically DTM++/dwr-fluid, especially by reusing the
 * datastructures, algorithms and/or supporting parameter/data input/output
 * classes, you are willing to cite the following three publications:
 * - J. Roth, J.P. Thiele, U. K&ouml;cher, T. Wick: "Tensor-Product Space-Time 
 *   Goal-Oriented Error Control and Adaptivity With Partition-of-Unity 
 *   Dual-Weighted Residuals for Nonstationary Flow Problems", 
 *   Computational Methods in Applied Mathematics, 2023.
 *   https://doi.org/10.1515/cmam-2022-0200
 *
 * - U. K&ouml;cher, M.P. Bruchh&auml;user, M. Bause: "Efficient and scalable
 *   data structures and algorithms for goal-oriented adaptivity of space-time
 *   FEM codes", SoftwareX 10(July-December):1-6, 100239, 2019.
 *
 * - U. K&ouml;cher: "Variational space-time methods for the elastic wave
 * equation and the diffusion equation", Ph.D. thesis, Department of Mechanical
 * Engineering of the Helmut-Schmidt-University, University of the German
 * Federal Armed Forces Hamburg, Germany, p. 1-188,
 *   urn:nbn:de:gbv:705-opus-31129, 2015. Open access via:
 *   http://edoc.sub.uni-hamburg.de/hsu/volltexte/2015/3112/
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher and contributors                   */
/*                                                                            */
/*  This file is part of DTM++.                                               */
/*                                                                            */
/*  DTM++ is free software: you can redistribute it and/or modify             */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++ is distributed in the hope that it will be useful,                  */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++.   If not, see <http://www.gnu.org/licenses/>.           */

// DEFINES

////////////////////////////////////////////////////////////////////////////////
// #define MPIX_THREADS 1
#define MPIX_THREADS 28
// #define MPIX_THREADS 4
// #define MPIX_THREADS dealii::numbers::invalid_unsigned_int
////////////////////////////////////////////////////////////////////////////////

// PROJECT includes
#include <DTM++/base/LogStream.hh>
#include <DTM++/base/Problem.hh>
#include <fluid/Fluid.tpl.hh>
#include <fluid/parameters/ParameterHandler.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

// C++ includes
#include <fstream>
#include <iostream>
#include <memory>

int main(int argc, char *argv[]) {
  // Init MPI (or MPI+X)
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, MPIX_THREADS);

  // EVALUATE wall time now.
  auto wall_time_start = MPI_Wtime();

  // Prepare DTM++ process logging to file
  DTM::pout.open();

  // Get MPI Variables
  const unsigned int MyPID(
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
  const unsigned int NumProc(
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

  //
  ////////////////////////////////////////////////////////////////////////////

  try {
    ////////////////////////////////////////////////////////////////////////
    // Init application
    //

    // Attach deallog to process output
    dealii::deallog.attach(DTM::pout);
    dealii::deallog.depth_console(0);
    DTM::pout << "****************************************"
              << "****************************************" << std::endl;

    DTM::pout << "DTM++/dwr-instatfluid" << std::endl
              << "Hej, here is process " << MyPID + 1 << " from " << NumProc
              << std::endl;

    // Check input arguments
    AssertThrow(!(argc < 2),
                dealii::ExcMessage(std::string(
                    "===>\tUSAGE: ./dwr-instatfluid <Input_Parameter_File.prm>")));

    // Check if the given input parameter file can be opened
    const std::string input_parameter_filename(argv[1]);
    {
      std::ifstream input_parameter_file(input_parameter_filename.c_str());
      AssertThrow(input_parameter_file,
                  dealii::ExcMessage(
                      std::string("===>\tERROR: Input parameter file <") +
                      input_parameter_filename + "> not found."));
    }

    // Prepare input parameter handling:
    auto parameter_handler = std::make_shared<fluid::ParameterHandler>();
    parameter_handler->parse_input(argv[1]);

    // Get minimal set of input parameters to drive the correct simulator
    unsigned int dimension;
    {
      // Problem dimension
      dimension =
          static_cast<unsigned int>(parameter_handler->get_integer("dim"));
    }

    //
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // Begin application
    //

    // create simulator
    std::shared_ptr<DTM::Problem> problem;

    // select simulator
    {
      DTM::pout << "dwr-instatfluid solver: primal cG(p)-dG(r)" << std::endl;

      switch (dimension) {
        case 2: {
          problem = std::make_shared<fluid::Fluid<2> >();
          break;
        }

        case 3: {
          problem = std::make_shared<fluid::Fluid<3> >();
          break;
        }

        default:
          dealii::ExcNotImplemented();
      }
    }

    DTM::pout << "dwr-fluid: dimension dim = " << dimension << std::endl
              << std::endl;

    DTM::pout << std::endl
              << "*********************************************************"
              << std::endl
              << std::endl;

    // run the simulator
    Assert(problem.use_count(), dealii::ExcNotInitialized());
    problem->set_input_parameters(parameter_handler);
    problem->run();

    DTM::pout << std::endl << "Goodbye." << std::endl;

    //
    // End application
    ////////////////////////////////////////////////////////////////////////////<
  } catch (std::exception &exc) {
    // EVALUATE program run time in terms of the consumed wall time.
    auto wall_time_end = MPI_Wtime();
    DTM::pout << std::endl
              << "Elapsed wall time: " << wall_time_end - wall_time_start
              << std::endl;

    if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
      std::cerr
          << std::endl
          << "****************************************"
          << "****************************************" << std::endl
          << std::endl
          << "An EXCEPTION occured: Please READ the following output CAREFULLY!"
          << std::endl;

      std::cerr << exc.what() << std::endl;

      std::cerr << std::endl
                << "APPLICATION TERMINATED unexpectedly due to an exception."
                << std::endl
                << std::endl
                << "****************************************"
                << "****************************************" << std::endl
                << std::endl;
    }

    // LOG error message to individual process output file.
    DTM::pout
        << std::endl
        << "****************************************"
        << "****************************************" << std::endl
        << std::endl
        << "An EXCEPTION occured: Please READ the following output CAREFULLY!"
        << std::endl;

    DTM::pout << exc.what() << std::endl;

    DTM::pout << std::endl
              << "APPLICATION TERMINATED unexpectedly due to an exception."
              << std::endl
              << std::endl
              << "****************************************"
              << "****************************************" << std::endl
              << std::endl;

    // Close output file stream
    DTM::pout.close();

    return 1;
  } catch (...) {
    // EVALUATE program run time in terms of the consumed wall time.
    auto wall_time_end = MPI_Wtime();
    DTM::pout << std::endl
              << "Elapsed wall time: " << wall_time_end - wall_time_start
              << std::endl;

    if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
      std::cerr << std::endl
                << "****************************************"
                << "****************************************" << std::endl
                << std::endl
                << "An UNKNOWN EXCEPTION occured!" << std::endl;

      std::cerr << std::endl
                << "----------------------------------------"
                << "----------------------------------------" << std::endl
                << std::endl
                << "Further information:" << std::endl
                << "\tThe main() function catched an exception" << std::endl
                << "\twhich is not inherited from std::exception." << std::endl
                << "\tYou have probably called 'throw' somewhere," << std::endl
                << "\tif you do not have done this, please contact the authors!"
                << std::endl
                << std::endl
                << "----------------------------------------"
                << "----------------------------------------" << std::endl;

      std::cerr << std::endl
                << "APPLICATION TERMINATED unexpectedly due to an exception."
                << std::endl
                << std::endl
                << "****************************************"
                << "****************************************" << std::endl
                << std::endl;
    }

    // LOG error message to individual process output file.
    DTM::pout << std::endl
              << "****************************************"
              << "****************************************" << std::endl
              << std::endl
              << "An UNKNOWN EXCEPTION occured!" << std::endl;

    DTM::pout << std::endl
              << "----------------------------------------"
              << "----------------------------------------" << std::endl
              << std::endl
              << "Further information:" << std::endl
              << "\tThe main() function catched an exception" << std::endl
              << "\twhich is not inherited from std::exception." << std::endl
              << "\tYou have probably called 'throw' somewhere," << std::endl
              << "\tif you do not have done this, please contact the authors!"
              << std::endl
              << std::endl
              << "----------------------------------------"
              << "----------------------------------------" << std::endl;

    DTM::pout << std::endl
              << "APPLICATION TERMINATED unexpectedly due to an exception."
              << std::endl
              << std::endl
              << "****************************************"
              << "****************************************" << std::endl
              << std::endl;

    // Close output file stream
    DTM::pout.close();

    return 1;
  }

  // EVALUATE program run time in terms of the consumed wall time.
  auto wall_time_end = MPI_Wtime();
  DTM::pout << std::endl
            << "Elapsed wall time: " << wall_time_end - wall_time_start
            << std::endl;

  // Close output file stream
  DTM::pout.close();

  return 0;
}
