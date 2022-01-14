/**
 * @file   FluidDataPostprocessor.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-18, Stokes, UK
 * @date 2019-01-18, eG, UK
 * @date 2018-06-20, deal.II v9.0, UK
 * @date 2017-10-05, Biot, UK
 * @date 2015-08-26, DTM++/awave, UK
 * @date 2015-02-24, DTM++/ccfd, UK
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher                                    */
/*                                                                            */
/*  This file is part of DTM++ .                                              */
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
/*  along with DTM++ .   If not, see <http://www.gnu.org/licenses/>.          */

#ifndef __FluidDataPostprocessor_tpl_hh
#define __FluidDataPostprocessor_tpl_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_postprocessor.h>

// C++ includes
#include <string>
#include <vector>


namespace fluid {

enum class OutputQuantities : unsigned int {
// 	displacement = 2<<0, ///< \f$ u \f$
// 	velocity     = 2<<1, ///< only for dynamic models
// 	stress       = 2<<2, ///< not implemented so far
// 	flux         = 2<<3, ///< \f$ q \f$
// 	div_flux     = 2<<4, ///< \f$ \nabla \cdot q \f$
	convection   = 2<<0,  ///< \f$ b \f$
	pressure     = 2<<1 ///< \f$ p \f$
};


/** Data post-processor for the solution of fluid.
 * 
 * More precisely, for the numerical solution vectors
 * \f$ x_h \f$
 * the following solution data components are extracted
 * 
 * - convection tensor field \f$ b_h \f$,
 * - pressure \f$ p_h \f$
 * 
 * with post-processing operations for an global output.
 * 
 * Objects from this class can directly attached to the deal.II DataOut write
 * functions and to the DTM++.core DTM::core::DataOutput write function.
 * 
 */
template<int dim>
class DataPostprocessor : public dealii::DataPostprocessor<dim> {
public:
	DataPostprocessor(
		const unsigned int output_quantities) :
		output_quantities{output_quantities},
		uflags{dealii::update_values} {
		
		////////////////////////////////////////////////////////////////////////
		// set up postprocessed component names and dci
		
		unsigned int internal_component=0;
		
		if (output_quantities & static_cast<unsigned int>(OutputQuantities::convection)) {
			internal_first_convection_component = internal_component;
			
			for (unsigned int d{0}; d < dim; ++d) {
				postprocessed_quantities_names.push_back("convection");
				
				dci.push_back(
					dealii::DataComponentInterpretation::component_is_part_of_vector
				);
				++internal_component;
			}
		}
		else {
			internal_first_convection_component = dealii::numbers::invalid_unsigned_int;
		}
		
		if (output_quantities & static_cast<unsigned int>(OutputQuantities::pressure)) {
			postprocessed_quantities_names.push_back("pressure");
			dci.push_back(dealii::DataComponentInterpretation::component_is_scalar);
			internal_pressure_component = internal_component;
			++internal_component;
		}
		else {
			internal_pressure_component = dealii::numbers::invalid_unsigned_int;
		}
		
		internal_n_components = internal_component;
		
		Assert(
			internal_n_components > 0,
			dealii::ExcMessage("You specified not any output quantity")
		);
	}
	
	virtual ~DataPostprocessor() = default;
	
	virtual
	void
	evaluate_vector_field(
		const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
		std::vector< dealii::Vector<double> > &postprocessed_quantities
	) const;
	
	virtual std::vector< std::string > get_names() const;
	
	virtual
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation >
	get_data_component_interpretation() const;
	
	virtual dealii::UpdateFlags get_needed_update_flags() const;
	
private:
	// physical quantities from FESystem
	const unsigned int first_convection_component = 0;
	const unsigned int pressure_component         = 1*dim;
	
	unsigned int output_quantities;
	
	// internal (derived) quantities from postprocessing
	unsigned int internal_first_convection_component;
	unsigned int internal_pressure_component;
	
	unsigned int internal_n_components;
	
	std::vector< std::string > postprocessed_quantities_names;
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci;
	
	dealii::UpdateFlags uflags;
};


} // namespace

#endif
