/**
 * @file TriaGenerator.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-09-27, JR
 * @date 2019-11-11, stokes, UK
 * @date 2018-07-26, included from biot/piot as TriaGenerator, UK
 * @date 2015-11-16, biot, UK
 * @date 2015-09-16, GridGenerator from input file, UK
 * @date 2015-05-15, UK
 * @date 2013-08-15, UK
 */

/*  Copyright (C) 2012-2021 by Uwe Koecher and contributors                   */
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

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// Project includes
#include <fluid/grid/TriaGenerator.tpl.hh>

// DEAL.II includes
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

// class declaration
namespace fluid {

/** Generates sophisticated triangulations.
 * <ul>
 * <li>hyper_cube: generates \f$ \Omega_h=\{ (p_1^1,p_2^1) \times (p_1^2,p_2^2)
 * \times (p_1^3,p_2^3) \} \f$ with one mesh cell; cf. deal.II documentation.</li>
 * 
 * <li>subdivided_hyper_rectangle: same as hyper_cube but allows for (an-)isotropic
 * refinement upon construction; cf. deal.II documentation.</li>
 * 
 * <li>hyper_ball: generates a circle for \f$ d=2 \f$ and a ball for \f$ d=3 \f$
 * for the given center (middle point) and radius. This deal.II grid_generator
 * automatically applies an boundary manifold for the (hyper-)sphere.</li>
 * 
 * <li>Lshape: generates \f$ \Omega_h=\{ (0,1)^d \} \setminus \{ (0.5,1)^d \} \f$,
 * \f$ d=2,3 \f$ with 3 mesh cells.</li>
 * </ul>
 * 
 * <li>backward facing step</li>
 * 
 * <li>constrained_channel</li>
 *
 * <li>channel_without_cylinder_2D: creates the grid for the 2D flow around a cylinder from Schaefer/Turek 1996</li>
 *
 * </ul>
 * 
 * This function can easily be extended to other triangulations/geometrical mesh
 * descriptions.
 */
template<int dim>
void
TriaGenerator<dim>::
generate(
	const std::string &TriaGenerator_Type,
	const std::string &TriaGenerator_Options,
	std::shared_ptr< dealii::Triangulation<dim> > tria ) {
	// check if tria is initialized
	Assert(
		tria.use_count(),
		dealii::ExcNotInitialized()
	);
	
	////////////////////////////////////////////////////////////////////////////
	// parse the input string, arguments are splitted with spaces
	//
	std::string argument;
	std::vector< std::string > options;
	for (auto &character : TriaGenerator_Options) {
		if (!std::isspace(character) && (character!='\"') ) {
			argument += character;
		}
		else {
			if (argument.size()) {
				options.push_back(argument);
				argument.clear();
			}
		}
	}
	
	if (argument.size()) {
		options.push_back(argument);
		argument.clear();
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("hyper_cube") == 0) {
		AssertThrow(
			options.size() == 2,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		dealii::GridGenerator::hyper_cube(
			*tria,
			std::stod(options.at(0)),
			std::stod(options.at(1))
		);
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("subdivided_hyper_rectangle") == 0) {
		AssertThrow(
			options.size() == 9,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		////////////////////////////////////////////////////////////////////////
		// Omega=(-x^1_val, x^1_val) x (-x^2_val, x^2_val) x (-x^3_val, x^3_val)
		//
		dealii::Point<dim> p1,p2;
		std::vector< unsigned int > repetitions(dim);
		{
			double x_i_val_a;
			double x_i_val_b;
			double x_i_h;
			
			for (unsigned int i(0); i < dim; ++i) {
				try {
					x_i_val_a = std::stod(options.at(i));
					x_i_val_b = std::stod(options.at(i+3));
					x_i_h = std::stod(options.at(i+6));
				}
				catch (std::invalid_argument &exc) {
					throw dealii::ExcMessage("TriaGenerator Options invalid, please check your input file data.");
				}
				catch (std::out_of_range &exc) {
					throw dealii::ExcMessage("TriaGenerator Options invalid, please check your input file data.");
				}
				
				// compute the repetitions for the grid:
				repetitions[i] = std::floor((x_i_val_b - x_i_val_a) / x_i_h);
				
				// set bottom left (p1) and upper right (p2) points
				p1[i] = x_i_val_a;
				p2[i] = x_i_val_b;
			}
		}
		
		dealii::GridGenerator::subdivided_hyper_rectangle(
			*tria,
			repetitions, p1, p2,
			false // colorize boundaries
		);
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("hyper_ball") == 0) {
		AssertThrow(
			options.size() == 4,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		dealii::Point<dim> center;
		for (unsigned int d{0}; d < dim; ++d) {
			center[d] = std::stod(options.at(d));
		}
		const double radius{std::stod(options.at(3))};
		Assert(radius>0, dealii::ExcMessage("radius must be > 0"));
		
		dealii::GridGenerator::hyper_ball(
			*tria,
			center,
			radius,
			false // attach_spherical_manifold_on_boundary_cells
		);
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("Lshape") == 0) {
		AssertThrow(
			((dim==2)||(dim==3)),
			dealii::ExcMessage("dim = 2,3 needed for Lshape Grid")
		);
		
		AssertThrow(
			options.size() == 0,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		Assert(tria.use_count(), dealii::ExcNotInitialized());
		
		{
			////////////////////////////////////////////////////////////////////
			std::vector< dealii::Point<dim> > vertices;
			dealii::Point<dim> node;
			
			if (dim == 2) {
				node[0]=0.0; node[1]=0.0; vertices.push_back(node); // 0
				node[0]=0.5; node[1]=0.0; vertices.push_back(node); // 1
				node[0]=1.0; node[1]=0.0; vertices.push_back(node); // 2
				
				node[0]=0.0; node[1]=0.5; vertices.push_back(node); // 3
				node[0]=0.5; node[1]=0.5; vertices.push_back(node); // 4
				node[0]=1.0; node[1]=0.5; vertices.push_back(node); // 5
				
				node[0]=0.0; node[1]=1.0; vertices.push_back(node); // 6
				node[0]=0.5; node[1]=1.0; vertices.push_back(node); // 7
			}
			else if (dim == 3) {
				node[2]=0.0;
				node[0]=0.0; node[1]=0.0; vertices.push_back(node); // 0
				node[0]=0.5; node[1]=0.0; vertices.push_back(node); // 1
				node[0]=1.0; node[1]=0.0; vertices.push_back(node); // 2
				
				node[0]=0.0; node[1]=0.5; vertices.push_back(node); // 3
				node[0]=0.5; node[1]=0.5; vertices.push_back(node); // 4
				node[0]=1.0; node[1]=0.5; vertices.push_back(node); // 5
				
				node[0]=0.0; node[1]=1.0; vertices.push_back(node); // 6
				node[0]=0.5; node[1]=1.0; vertices.push_back(node); // 7
				
				node[2]=0.5;
				node[0]=0.0; node[1]=0.0; vertices.push_back(node); // 8
				node[0]=0.5; node[1]=0.0; vertices.push_back(node); // 9
				node[0]=1.0; node[1]=0.0; vertices.push_back(node); // 10
				
				node[0]=0.0; node[1]=0.5; vertices.push_back(node); // 11
				node[0]=0.5; node[1]=0.5; vertices.push_back(node); // 12
				node[0]=1.0; node[1]=0.5; vertices.push_back(node); // 13
				
				node[0]=0.0; node[1]=1.0; vertices.push_back(node); // 14
				node[0]=0.5; node[1]=1.0; vertices.push_back(node); // 15
			}
			
			////////////////////////////////////////////////////////////////////
			std::vector< std::vector<unsigned int> > cell_vertices;
			
			if (dim == 2) {
				std::vector<unsigned int> cell_indices = {0,0,0,0};
				
				cell_indices = {0, 1, 3, 4};
				cell_vertices.push_back(cell_indices);
				
				cell_indices = {1, 2, 4, 5};
				cell_vertices.push_back(cell_indices);
				
				cell_indices = {3, 4, 6, 7};
				cell_vertices.push_back(cell_indices);
			}
			
			if (dim == 3) {
				std::vector<unsigned int> cell_indices = {0,0,0,0, 0,0,0,0};
				
				cell_indices = {0, 1, 3, 4, 0+8, 1+8, 3+8, 4+8};
				cell_vertices.push_back(cell_indices);
				
				cell_indices = {1, 2, 4, 5, 1+8, 2+8, 4+8, 5+8};
				cell_vertices.push_back(cell_indices);
				
				cell_indices = {3, 4, 6, 7, 3+8, 4+8, 6+8, 7+8};
				cell_vertices.push_back(cell_indices);
			}
			
			////////////////////////////////////////////////////////////////////
			std::vector< dealii::CellData<dim> > cells;
			cells.resize( cell_vertices.size() );
			
			for (unsigned int i=0; i < cells.size(); ++i) {
				for (unsigned int j=0; j < dealii::GeometryInfo<dim>::vertices_per_cell; ++j) {
					cells[i].vertices[j] = cell_vertices[i][j];
				}
			}
			
			tria->create_triangulation(vertices, cells, dealii::SubCellData());
		}
		
		DTM::pout << "DTM++: TriaGenerator: Successfully created Lshape grid."
			<< std::endl;
		
		DTM::pout << "\tn_global_active_cells = "
			<< tria->n_global_active_cells()
			<< std::endl << std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("backward_facing_step") == 0) {
		AssertThrow(
			((dim==2) || (dim==3)),
			dealii::ExcMessage("dim = 2,3 needed for backward facing step Grid")
		);
		
		AssertThrow(
			options.size() == 12,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		////////////////////////////////////////////////////////////////////
		// create mesh geometry one rank=0 proc and write out the mesh file
		//
		if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
			dealii::Triangulation<dim> backward_facing_step_tria;
			
			////////////////////////////////////////////////////////////////////////
			// Omega=(-x^1_val, x^1_val) x (-x^2_val, x^2_val) x (-x^3_val, x^3_val)
			// + step
			
			// create channel
			dealii::Point<dim> p1,p2;
			std::vector< unsigned int > repetitions(dim);
			{
				double x_i_val_a;
				double x_i_val_b;
				double x_i_h;
				
				for (unsigned int i(0); i < dim; ++i) {
					try {
						x_i_val_a = std::stod(options.at(i));
						x_i_val_b = std::stod(options.at(i+3));
						x_i_h = std::stod(options.at(i+6));
					}
					catch (std::invalid_argument &exc) {
						throw dealii::ExcMessage("TriaGenerator Options invalid, please check your input file data.");
					}
					catch (std::out_of_range &exc) {
						throw dealii::ExcMessage("TriaGenerator Options invalid, please check your input file data.");
					}
					
					// compute the repetitions for the grid:
					repetitions[i] = std::floor((x_i_val_b - x_i_val_a) / x_i_h);
					
					// set bottom left (p1) and upper right (p2) points
					p1[i] = x_i_val_a;
					p2[i] = x_i_val_b;
				}
			}
			
			dealii::Triangulation<dim> channel_tria;
			dealii::GridGenerator::subdivided_hyper_rectangle(
				channel_tria,
				repetitions, p1, p2,
				false // colorize boundaries
			);
			
			// create step
			const double step_length_x = std::stod(options.at(9));
			const double step_length_y = std::stod(options.at(10));
			const double step_length_z = std::stod(options.at(11));
			
			dealii::Point<dim> ps1,ps2;
			if (dim==2) {
				ps2[0] = p1[0];
				ps2[1] = p2[1];
				
				ps1[0] = ps2[0]-step_length_x;
				ps1[1] = ps2[1]-step_length_y;
			}
			else {
				// dim==3
				ps2[0] = p2[0];
				ps2[1] = p1[1];
				ps2[2] = p2[2];
				
				ps1[0] = ps2[0]-step_length_x;
				ps1[1] = ps2[1]-step_length_y;
				ps1[1] = ps2[1]-step_length_z;
			}
			
			dealii::Triangulation<dim> step_tria;
			dealii::GridGenerator::hyper_rectangle(
				step_tria,
				ps1,
				ps2,
				false
			);
			
			// merge step and channel
			dealii::GridGenerator::merge_triangulations(
				step_tria, channel_tria,
				backward_facing_step_tria
			);
			
			// write to file
			dealii::GridOut grid_out;
			std::ofstream output_msh("backward_facing_step.msh", std::ios_base::out);
			grid_out.write_msh(backward_facing_step_tria, output_msh);
			output_msh.close();
			
			////////////////////////////////////////////////////////////////////
		}
		
		Assert(tria.use_count(), dealii::ExcNotInitialized());
		
		////////////////////////////////////////////////////////////////////
		//  Grid In
		{
			dealii::GridIn<dim> grid_in;
			grid_in.attach_triangulation(*tria);
			std::ifstream input_msh("backward_facing_step.msh", std::ios_base::in);
			grid_in.read_msh(input_msh);
			input_msh.close();
		}
		
		DTM::pout << "DTM++: TriaGenerator: Successfully created backward facing step grid."
			<< std::endl;
		
		DTM::pout << "\tn_global_active_cells = "
			<< tria->n_global_active_cells()
			<< std::endl << std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("constrained_channel") == 0) {
		AssertThrow(
			(dim==2),
			dealii::ExcMessage("dim = 2 needed for constrained channel TriaGenerator")
		);
		
		AssertThrow(
			options.size() == 9*3,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		////////////////////////////////////////////////////////////////////
		// create mesh geometry one rank=0 proc and write out the mesh file
		//
		if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
			dealii::Triangulation<dim> tria_tmp1;
			dealii::Triangulation<dim> tria_tmp2;
			
			dealii::Point<dim> p1,p2;
			std::vector< unsigned int > repetitions(dim);
			
			// create inflow channel (left)
			{
				double x_i_val_a;
				double x_i_val_b;
				double x_i_h;
				
				for (unsigned int i(0); i < dim; ++i) {
					x_i_val_a = std::stod(options.at(i));
					x_i_val_b = std::stod(options.at(i+3));
					x_i_h = std::stod(options.at(i+6));
					
					// compute the repetitions for the grid:
					repetitions[i] = std::floor((x_i_val_b - x_i_val_a) / x_i_h);
					
					// set bottom left (p1) and upper right (p2) points
					p1[i] = x_i_val_a;
					p2[i] = x_i_val_b;
				}
			}
			
			dealii::Triangulation<dim> inflow_channel;
			dealii::GridGenerator::subdivided_hyper_rectangle(
				inflow_channel,
				repetitions, p1, p2,
				false // colorize boundaries
			);
			
			// create constrained channel (middle)
			{
				double x_i_val_a;
				double x_i_val_b;
				double x_i_h;
				
				for (unsigned int i(0); i < dim; ++i) {
					x_i_val_a = std::stod(options.at(i+9));
					x_i_val_b = std::stod(options.at(i+3+9));
					x_i_h = std::stod(options.at(i+6+9));
					
					// compute the repetitions for the grid:
					repetitions[i] = std::floor((x_i_val_b - x_i_val_a) / x_i_h);
					
					// set bottom left (p1) and upper right (p2) points
					p1[i] = x_i_val_a;
					p2[i] = x_i_val_b;
				}
			}
			
			dealii::Triangulation<dim> constrained_channel;
			dealii::GridGenerator::subdivided_hyper_rectangle(
				constrained_channel,
				repetitions, p1, p2,
				false // colorize boundaries
			);
			
			// create outflow channel (right)
			{
				double x_i_val_a;
				double x_i_val_b;
				double x_i_h;
				
				for (unsigned int i(0); i < dim; ++i) {
					x_i_val_a = std::stod(options.at(i+9*2));
					x_i_val_b = std::stod(options.at(i+3+9*2));
					x_i_h = std::stod(options.at(i+6+9*2));
					
					// compute the repetitions for the grid:
					repetitions[i] = std::floor((x_i_val_b - x_i_val_a) / x_i_h);
					
					// set bottom left (p1) and upper right (p2) points
					p1[i] = x_i_val_a;
					p2[i] = x_i_val_b;
				}
			}
			
			dealii::Triangulation<dim> outflow_channel;
			dealii::GridGenerator::subdivided_hyper_rectangle(
				outflow_channel,
				repetitions, p1, p2,
				false // colorize boundaries
			);
			
			
			// merge
			dealii::GridGenerator::merge_triangulations(
				inflow_channel, constrained_channel,
				tria_tmp1
			);
			
			dealii::GridGenerator::merge_triangulations(
				tria_tmp1, outflow_channel,
				tria_tmp2
			);
			
			// write to file
			dealii::GridOut grid_out;
			std::ofstream output_msh("constrained_channel.msh", std::ios_base::out);
			grid_out.write_msh(tria_tmp2, output_msh);
			output_msh.close();
			
// 			// sync all processes
// 			MPI_Barrier(Grid<dim>::mpi_comm);
			////////////////////////////////////////////////////////////////////
		}
		
		Assert(tria.use_count(), dealii::ExcNotInitialized());
		
		////////////////////////////////////////////////////////////////////
		//  Grid In
		{
			dealii::GridIn<dim> grid_in;
			grid_in.attach_triangulation(*tria);
			std::ifstream input_msh("constrained_channel.msh", std::ios_base::in);
			grid_in.read_msh(input_msh);
			input_msh.close();
		}
		
		DTM::pout << "DTM++: TriaGenerator: Successfully created constrained channel with TriaGenerator."
			<< std::endl;
		
		DTM::pout << "\tn_global_active_cells = "
			<< tria->n_global_active_cells()
			<< std::endl << std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("channel_without_cylinder_2D") == 0) {
		AssertThrow(
			options.size() == 0,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);

        std::string grid_name = "/home/ifam/thiele/grids/nsbench4_original.inp";
        dealii::GridIn<dim> grid_in;
        grid_in.attach_triangulation(*tria);
        std::ifstream input_file(grid_name.c_str());
        grid_in.read_ucd(input_file);
        dealii::Point<dim> p(0.2,0.2);
        static const dealii::SphericalManifold<dim> boundary(p);
        dealii::GridTools::copy_boundary_to_manifold_id(*tria);
        tria->set_manifold(80,boundary);

		return;
	}


	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("channel_without_cylinder_3D") == 0) {
		AssertThrow(
			options.size() == 0,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);

		std::string grid_name = "/home/ifam/thiele/grids/nsbench3dZ_3D.inp";
		dealii::GridIn<dim> grid_in;
		grid_in.attach_triangulation(*tria);
		std::ifstream input_file(grid_name.c_str());
		grid_in.read_ucd(input_file);
		dealii::Point<dim> p(0.5,0.2,0.);
		dealii::Tensor<1,dim> z_dir;
		z_dir[0] = 0.;
		z_dir[1] = 0.;
		z_dir[dim-1] = 1.;
		static const dealii::CylindricalManifold<dim> boundary(z_dir,p);
		dealii::GridTools::copy_boundary_to_manifold_id(*tria);
		tria->set_manifold(80,boundary);

		return;
	}
	////////////////////////////////////////////////////////////////////////////
	//
	AssertThrow(
		false,
		dealii::ExcMessage("TriaGenerator_Type unknown, please check your input file data.")
	);
	
}

} // namespaces

#include "TriaGenerator.inst.in"
