/**
 * @file Force_Selector.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-07, UK
 * @date 2018-07-30, dwr, MPB
 * @date 2018-07-26, dwr, UK
 * @date 2018-05-28, piot/ewave, UK
 * @date 2016-05-30, UK
 * @date 2016-02-11, UK
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

#include <DTM++/base/LogStream.hh>

#include <fluid/Force/Force_Selector.tpl.hh>
#include <fluid/Force/FluidForces.hh>

// C++ includes
#include <vector>

namespace fluid {
namespace force {

template<int dim>
void
Selector<dim>::
create_function(
		const std::string &_type,
		const std::string &_options,
		std::shared_ptr< dealii::TensorFunction<1,dim> > &tensor_function
	) const {
	
	////////////////////////////////////////////////////////////////////////////
	// parse the input string, arguments are splitted with spaces
	//
	std::string argument;
	std::vector< std::string > options;
	for (auto &character : _options) {
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
	
	DTM::pout << "* found configuration: fluid force function = " << _type << std::endl;
	DTM::pout << "* found configuration: fluid force options = " << std::endl;
	for (auto &option : options) {
		DTM::pout << "\t" << option << std::endl;
	}
	DTM::pout << std::endl;
	
	DTM::pout << "* generating function" << std::endl;
	
	if (_type.compare("ZeroTensorFunction") == 0) {
		AssertThrow(
			options.size() == 0,
			dealii::ExcMessage(
				"fluid force options invalid, "
				"please check your input file data."
			)
		);
		
		tensor_function =
			std::make_shared< dealii::ZeroTensorFunction<1,dim,double> > ();
		
		DTM::pout
			<< "fluid force selector: created zero tensor function" << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ConstantTensorFunction") == 0) {
		AssertThrow(
			options.size() == 3,
			dealii::ExcMessage(
				"fluid force options invalid, "
				"please check your input file data."
			)
		);
		
		dealii::Tensor<1, dim, double> f;
		for (unsigned int c={0}; c < dim; ++c) {
			f[c] = std::stod(options.at(c));
		}
		
		tensor_function = std::make_shared<
			dealii::ConstantTensorFunction<1,dim,double> > (
			f
		);
		
		DTM::pout
			<< "fluid force selector: created ConstantTensorFunction "
			<< "as force function, with " << std::endl
			<< "\tf(1) = " << std::stod(options.at(0)) << " , " << std::endl
			<< "\tf(2) = " << std::stod(options.at(1)) << " , " << std::endl
			<< "\tf(3) = " << std::stod(options.at(2)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	AssertThrow(
		false,
		dealii::ExcMessage("fluid force function unknown, please check your input file data.")
	);
}

}} //namespaces

#include "Force_Selector.inst.in"
