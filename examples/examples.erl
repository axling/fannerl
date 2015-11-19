%% Fannerl: Erlang Bindings to the Fast Artificial Neural Network Library (fann)
%% Copyright (C) 2015 Erik Axling (erik.axling@gmail.com)

%% This library is free software; you can redistribute it and/or
%% modify it under the terms of the GNU Lesser General Public
%% License as published by the Free Software Foundation; either
%% version 2.1 of the License, or (at your option) any later version.

%% This library is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%% Lesser General Public License for more details.

%% You should have received a copy of the GNU Lesser General Public
%% License along with this library; if not, write to the Free Software
%% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

%% The fannerl version of FANNs mushroom.c example
-module(examples).

-export([start/0]).

start() ->
    io:format("Start examples~n", []),
    io:format("~nExample: simple~n", []),
    simple:run(),
    io:format("~nExample: robot~n", []),
    robot:run(),
    io:format("~nExample: mushroom~n", []),
    mushroom:run(),
    io:format("~nExample: steepness_train~n", []),
    steepness_train:run(),
    io:format("~nExamples done~n", []),
    halt(0).
    
