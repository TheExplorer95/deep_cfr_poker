??"
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
~
dense_234/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_234/kernel
w
$dense_234/kernel/Read/ReadVariableOpReadVariableOpdense_234/kernel* 
_output_shapes
:
??*
dtype0
u
dense_234/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_234/bias
n
"dense_234/bias/Read/ReadVariableOpReadVariableOpdense_234/bias*
_output_shapes	
:?*
dtype0
~
dense_235/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_235/kernel
w
$dense_235/kernel/Read/ReadVariableOpReadVariableOpdense_235/kernel* 
_output_shapes
:
??*
dtype0
u
dense_235/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_235/bias
n
"dense_235/bias/Read/ReadVariableOpReadVariableOpdense_235/bias*
_output_shapes	
:?*
dtype0
}
dense_237/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_237/kernel
v
$dense_237/kernel/Read/ReadVariableOpReadVariableOpdense_237/kernel*
_output_shapes
:	?*
dtype0
u
dense_237/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_237/bias
n
"dense_237/bias/Read/ReadVariableOpReadVariableOpdense_237/bias*
_output_shapes	
:?*
dtype0
~
dense_236/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_236/kernel
w
$dense_236/kernel/Read/ReadVariableOpReadVariableOpdense_236/kernel* 
_output_shapes
:
??*
dtype0
u
dense_236/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_236/bias
n
"dense_236/bias/Read/ReadVariableOpReadVariableOpdense_236/bias*
_output_shapes	
:?*
dtype0
~
dense_238/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_238/kernel
w
$dense_238/kernel/Read/ReadVariableOpReadVariableOpdense_238/kernel* 
_output_shapes
:
??*
dtype0
u
dense_238/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_238/bias
n
"dense_238/bias/Read/ReadVariableOpReadVariableOpdense_238/bias*
_output_shapes	
:?*
dtype0
~
dense_239/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_239/kernel
w
$dense_239/kernel/Read/ReadVariableOpReadVariableOpdense_239/kernel* 
_output_shapes
:
??*
dtype0
u
dense_239/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_239/bias
n
"dense_239/bias/Read/ReadVariableOpReadVariableOpdense_239/bias*
_output_shapes	
:?*
dtype0
~
dense_240/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_240/kernel
w
$dense_240/kernel/Read/ReadVariableOpReadVariableOpdense_240/kernel* 
_output_shapes
:
??*
dtype0
u
dense_240/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_240/bias
n
"dense_240/bias/Read/ReadVariableOpReadVariableOpdense_240/bias*
_output_shapes	
:?*
dtype0
~
dense_241/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_241/kernel
w
$dense_241/kernel/Read/ReadVariableOpReadVariableOpdense_241/kernel* 
_output_shapes
:
??*
dtype0
u
dense_241/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_241/bias
n
"dense_241/bias/Read/ReadVariableOpReadVariableOpdense_241/bias*
_output_shapes	
:?*
dtype0
}
dense_242/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_242/kernel
v
$dense_242/kernel/Read/ReadVariableOpReadVariableOpdense_242/kernel*
_output_shapes
:	?*
dtype0
t
dense_242/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_242/bias
m
"dense_242/bias/Read/ReadVariableOpReadVariableOpdense_242/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
embedding_158/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*)
shared_nameembedding_158/embeddings
?
,embedding_158/embeddings/Read/ReadVariableOpReadVariableOpembedding_158/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_156/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameembedding_156/embeddings
?
,embedding_156/embeddings/Read/ReadVariableOpReadVariableOpembedding_156/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_157/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameembedding_157/embeddings
?
,embedding_157/embeddings/Read/ReadVariableOpReadVariableOpembedding_157/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_161/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*)
shared_nameembedding_161/embeddings
?
,embedding_161/embeddings/Read/ReadVariableOpReadVariableOpembedding_161/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_159/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameembedding_159/embeddings
?
,embedding_159/embeddings/Read/ReadVariableOpReadVariableOpembedding_159/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_160/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameembedding_160/embeddings
?
,embedding_160/embeddings/Read/ReadVariableOpReadVariableOpembedding_160/embeddings*
_output_shapes
:	?*
dtype0
?
"normalize_26/normalization_26/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"normalize_26/normalization_26/mean
?
6normalize_26/normalization_26/mean/Read/ReadVariableOpReadVariableOp"normalize_26/normalization_26/mean*
_output_shapes	
:?*
dtype0
?
&normalize_26/normalization_26/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&normalize_26/normalization_26/variance
?
:normalize_26/normalization_26/variance/Read/ReadVariableOpReadVariableOp&normalize_26/normalization_26/variance*
_output_shapes	
:?*
dtype0
?
#normalize_26/normalization_26/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *4
shared_name%#normalize_26/normalization_26/count
?
7normalize_26/normalization_26/count/Read/ReadVariableOpReadVariableOp#normalize_26/normalization_26/count*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_234/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_234/kernel/m
?
+Adam/dense_234/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_234/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_234/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_234/bias/m
|
)Adam/dense_234/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_234/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_235/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_235/kernel/m
?
+Adam/dense_235/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_235/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_235/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_235/bias/m
|
)Adam/dense_235/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_235/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_237/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_237/kernel/m
?
+Adam/dense_237/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_237/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_237/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_237/bias/m
|
)Adam/dense_237/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_237/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_236/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_236/kernel/m
?
+Adam/dense_236/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_236/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_236/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_236/bias/m
|
)Adam/dense_236/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_236/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_238/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_238/kernel/m
?
+Adam/dense_238/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_238/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_238/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_238/bias/m
|
)Adam/dense_238/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_238/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_239/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_239/kernel/m
?
+Adam/dense_239/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_239/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_239/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_239/bias/m
|
)Adam/dense_239/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_239/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_240/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_240/kernel/m
?
+Adam/dense_240/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_240/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_240/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_240/bias/m
|
)Adam/dense_240/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_240/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_241/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_241/kernel/m
?
+Adam/dense_241/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_241/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_241/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_241/bias/m
|
)Adam/dense_241/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_241/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_242/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_242/kernel/m
?
+Adam/dense_242/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_242/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_242/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_242/bias/m
{
)Adam/dense_242/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_242/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_158/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*0
shared_name!Adam/embedding_158/embeddings/m
?
3Adam/embedding_158/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_158/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_156/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_156/embeddings/m
?
3Adam/embedding_156/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_156/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_157/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_157/embeddings/m
?
3Adam/embedding_157/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_157/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_161/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*0
shared_name!Adam/embedding_161/embeddings/m
?
3Adam/embedding_161/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_161/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_159/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_159/embeddings/m
?
3Adam/embedding_159/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_159/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_160/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_160/embeddings/m
?
3Adam/embedding_160/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_160/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_234/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_234/kernel/v
?
+Adam/dense_234/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_234/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_234/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_234/bias/v
|
)Adam/dense_234/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_234/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_235/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_235/kernel/v
?
+Adam/dense_235/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_235/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_235/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_235/bias/v
|
)Adam/dense_235/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_235/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_237/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_237/kernel/v
?
+Adam/dense_237/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_237/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_237/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_237/bias/v
|
)Adam/dense_237/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_237/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_236/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_236/kernel/v
?
+Adam/dense_236/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_236/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_236/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_236/bias/v
|
)Adam/dense_236/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_236/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_238/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_238/kernel/v
?
+Adam/dense_238/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_238/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_238/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_238/bias/v
|
)Adam/dense_238/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_238/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_239/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_239/kernel/v
?
+Adam/dense_239/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_239/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_239/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_239/bias/v
|
)Adam/dense_239/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_239/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_240/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_240/kernel/v
?
+Adam/dense_240/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_240/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_240/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_240/bias/v
|
)Adam/dense_240/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_240/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_241/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_241/kernel/v
?
+Adam/dense_241/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_241/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_241/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_241/bias/v
|
)Adam/dense_241/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_241/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_242/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_242/kernel/v
?
+Adam/dense_242/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_242/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_242/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_242/bias/v
{
)Adam/dense_242/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_242/bias/v*
_output_shapes
:*
dtype0
?
Adam/embedding_158/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*0
shared_name!Adam/embedding_158/embeddings/v
?
3Adam/embedding_158/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_158/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_156/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_156/embeddings/v
?
3Adam/embedding_156/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_156/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_157/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_157/embeddings/v
?
3Adam/embedding_157/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_157/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_161/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*0
shared_name!Adam/embedding_161/embeddings/v
?
3Adam/embedding_161/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_161/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_159/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_159/embeddings/v
?
3Adam/embedding_159/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_159/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_160/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/embedding_160/embeddings/v
?
3Adam/embedding_160/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_160/embeddings/v*
_output_shapes
:	?*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 * $tI
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
??
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
trainable_variables
	variables
regularization_losses
 	keras_api
!
signatures
 
 
 
?
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api

R	keras_api
h

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
h

_kernel
`bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
h

ekernel
fbias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api

k	keras_api
h

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api

r	keras_api
h

skernel
tbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api

?	keras_api

?	keras_api
f
?	normalize
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26
 
?
?layers
trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2

?0
?1
?2
 
?
?layers
1trainable_variables
 ?layer_regularization_losses
2	variables
3regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2

?0
?1
?2
 
?
?layers
Dtrainable_variables
 ?layer_regularization_losses
E	variables
Fregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 
 
\Z
VARIABLE_VALUEdense_234/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_234/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
?
?layers
Ntrainable_variables
 ?layer_regularization_losses
O	variables
Pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
\Z
VARIABLE_VALUEdense_235/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_235/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
?layers
Utrainable_variables
 ?layer_regularization_losses
V	variables
Wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
\Z
VARIABLE_VALUEdense_237/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_237/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
?
?layers
[trainable_variables
 ?layer_regularization_losses
\	variables
]regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
\Z
VARIABLE_VALUEdense_236/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_236/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
 
?
?layers
atrainable_variables
 ?layer_regularization_losses
b	variables
cregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
\Z
VARIABLE_VALUEdense_238/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_238/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
?
?layers
gtrainable_variables
 ?layer_regularization_losses
h	variables
iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
\Z
VARIABLE_VALUEdense_239/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_239/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
?
?layers
ntrainable_variables
 ?layer_regularization_losses
o	variables
pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
\Z
VARIABLE_VALUEdense_240/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_240/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
?
?layers
utrainable_variables
 ?layer_regularization_losses
v	variables
wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
\Z
VARIABLE_VALUEdense_241/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_241/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
?
?layers
}trainable_variables
 ?layer_regularization_losses
~	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
c
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api
 

?0
?1
?2
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
][
VARIABLE_VALUEdense_242/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_242/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEembedding_158/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEembedding_156/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEembedding_157/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEembedding_161/embeddings0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEembedding_159/embeddings0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEembedding_160/embeddings0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"normalize_26/normalization_26/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&normalize_26/normalization_26/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#normalize_26/normalization_26/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 
 

?0

?0
?1
?2
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 
 
 
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
 
 
 
 
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 
 
 
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
&
	?mean
?variance

?count
 
 

?0
 
 
 

?0
?1
?2
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
}
VARIABLE_VALUEAdam/dense_234/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_234/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_235/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_235/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_237/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_237/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_236/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_236/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_238/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_238/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_239/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_239/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_240/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_240/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_241/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_241/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_242/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_242/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_158/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_156/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_157/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_161/embeddings/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_159/embeddings/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_160/embeddings/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_234/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_234/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_235/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_235/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_237/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_237/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_236/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_236/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_238/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_238/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_239/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_239/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_240/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_240/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_241/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_241/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_242/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_242/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_158/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_156/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_157/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_161/embeddings/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_159/embeddings/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/embedding_160/embeddings/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_betsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

y
serving_default_cards0Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_cards1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_158/embeddingsembedding_156/embeddingsembedding_157/embeddingsConst_2embedding_161/embeddingsembedding_159/embeddingsembedding_160/embeddingsConst_3Const_4dense_234/kerneldense_234/biasdense_237/kerneldense_237/biasdense_235/kerneldense_235/biasdense_236/kerneldense_236/biasdense_238/kerneldense_238/biasdense_239/kerneldense_239/biasdense_240/kerneldense_240/biasdense_241/kerneldense_241/bias"normalize_26/normalization_26/mean&normalize_26/normalization_26/variancedense_242/kerneldense_242/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *0
f+R)
'__inference_signature_wrapper_256360498
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_234/kernel/Read/ReadVariableOp"dense_234/bias/Read/ReadVariableOp$dense_235/kernel/Read/ReadVariableOp"dense_235/bias/Read/ReadVariableOp$dense_237/kernel/Read/ReadVariableOp"dense_237/bias/Read/ReadVariableOp$dense_236/kernel/Read/ReadVariableOp"dense_236/bias/Read/ReadVariableOp$dense_238/kernel/Read/ReadVariableOp"dense_238/bias/Read/ReadVariableOp$dense_239/kernel/Read/ReadVariableOp"dense_239/bias/Read/ReadVariableOp$dense_240/kernel/Read/ReadVariableOp"dense_240/bias/Read/ReadVariableOp$dense_241/kernel/Read/ReadVariableOp"dense_241/bias/Read/ReadVariableOp$dense_242/kernel/Read/ReadVariableOp"dense_242/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,embedding_158/embeddings/Read/ReadVariableOp,embedding_156/embeddings/Read/ReadVariableOp,embedding_157/embeddings/Read/ReadVariableOp,embedding_161/embeddings/Read/ReadVariableOp,embedding_159/embeddings/Read/ReadVariableOp,embedding_160/embeddings/Read/ReadVariableOp6normalize_26/normalization_26/mean/Read/ReadVariableOp:normalize_26/normalization_26/variance/Read/ReadVariableOp7normalize_26/normalization_26/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_234/kernel/m/Read/ReadVariableOp)Adam/dense_234/bias/m/Read/ReadVariableOp+Adam/dense_235/kernel/m/Read/ReadVariableOp)Adam/dense_235/bias/m/Read/ReadVariableOp+Adam/dense_237/kernel/m/Read/ReadVariableOp)Adam/dense_237/bias/m/Read/ReadVariableOp+Adam/dense_236/kernel/m/Read/ReadVariableOp)Adam/dense_236/bias/m/Read/ReadVariableOp+Adam/dense_238/kernel/m/Read/ReadVariableOp)Adam/dense_238/bias/m/Read/ReadVariableOp+Adam/dense_239/kernel/m/Read/ReadVariableOp)Adam/dense_239/bias/m/Read/ReadVariableOp+Adam/dense_240/kernel/m/Read/ReadVariableOp)Adam/dense_240/bias/m/Read/ReadVariableOp+Adam/dense_241/kernel/m/Read/ReadVariableOp)Adam/dense_241/bias/m/Read/ReadVariableOp+Adam/dense_242/kernel/m/Read/ReadVariableOp)Adam/dense_242/bias/m/Read/ReadVariableOp3Adam/embedding_158/embeddings/m/Read/ReadVariableOp3Adam/embedding_156/embeddings/m/Read/ReadVariableOp3Adam/embedding_157/embeddings/m/Read/ReadVariableOp3Adam/embedding_161/embeddings/m/Read/ReadVariableOp3Adam/embedding_159/embeddings/m/Read/ReadVariableOp3Adam/embedding_160/embeddings/m/Read/ReadVariableOp+Adam/dense_234/kernel/v/Read/ReadVariableOp)Adam/dense_234/bias/v/Read/ReadVariableOp+Adam/dense_235/kernel/v/Read/ReadVariableOp)Adam/dense_235/bias/v/Read/ReadVariableOp+Adam/dense_237/kernel/v/Read/ReadVariableOp)Adam/dense_237/bias/v/Read/ReadVariableOp+Adam/dense_236/kernel/v/Read/ReadVariableOp)Adam/dense_236/bias/v/Read/ReadVariableOp+Adam/dense_238/kernel/v/Read/ReadVariableOp)Adam/dense_238/bias/v/Read/ReadVariableOp+Adam/dense_239/kernel/v/Read/ReadVariableOp)Adam/dense_239/bias/v/Read/ReadVariableOp+Adam/dense_240/kernel/v/Read/ReadVariableOp)Adam/dense_240/bias/v/Read/ReadVariableOp+Adam/dense_241/kernel/v/Read/ReadVariableOp)Adam/dense_241/bias/v/Read/ReadVariableOp+Adam/dense_242/kernel/v/Read/ReadVariableOp)Adam/dense_242/bias/v/Read/ReadVariableOp3Adam/embedding_158/embeddings/v/Read/ReadVariableOp3Adam/embedding_156/embeddings/v/Read/ReadVariableOp3Adam/embedding_157/embeddings/v/Read/ReadVariableOp3Adam/embedding_161/embeddings/v/Read/ReadVariableOp3Adam/embedding_159/embeddings/v/Read/ReadVariableOp3Adam/embedding_160/embeddings/v/Read/ReadVariableOpConst_5*_
TinX
V2T		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__traced_save_256361796
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_234/kerneldense_234/biasdense_235/kerneldense_235/biasdense_237/kerneldense_237/biasdense_236/kerneldense_236/biasdense_238/kerneldense_238/biasdense_239/kerneldense_239/biasdense_240/kerneldense_240/biasdense_241/kerneldense_241/biasdense_242/kerneldense_242/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_158/embeddingsembedding_156/embeddingsembedding_157/embeddingsembedding_161/embeddingsembedding_159/embeddingsembedding_160/embeddings"normalize_26/normalization_26/mean&normalize_26/normalization_26/variance#normalize_26/normalization_26/counttotalcountAdam/dense_234/kernel/mAdam/dense_234/bias/mAdam/dense_235/kernel/mAdam/dense_235/bias/mAdam/dense_237/kernel/mAdam/dense_237/bias/mAdam/dense_236/kernel/mAdam/dense_236/bias/mAdam/dense_238/kernel/mAdam/dense_238/bias/mAdam/dense_239/kernel/mAdam/dense_239/bias/mAdam/dense_240/kernel/mAdam/dense_240/bias/mAdam/dense_241/kernel/mAdam/dense_241/bias/mAdam/dense_242/kernel/mAdam/dense_242/bias/mAdam/embedding_158/embeddings/mAdam/embedding_156/embeddings/mAdam/embedding_157/embeddings/mAdam/embedding_161/embeddings/mAdam/embedding_159/embeddings/mAdam/embedding_160/embeddings/mAdam/dense_234/kernel/vAdam/dense_234/bias/vAdam/dense_235/kernel/vAdam/dense_235/bias/vAdam/dense_237/kernel/vAdam/dense_237/bias/vAdam/dense_236/kernel/vAdam/dense_236/bias/vAdam/dense_238/kernel/vAdam/dense_238/bias/vAdam/dense_239/kernel/vAdam/dense_239/bias/vAdam/dense_240/kernel/vAdam/dense_240/bias/vAdam/dense_241/kernel/vAdam/dense_241/bias/vAdam/dense_242/kernel/vAdam/dense_242/bias/vAdam/embedding_158/embeddings/vAdam/embedding_156/embeddings/vAdam/embedding_157/embeddings/vAdam/embedding_161/embeddings/vAdam/embedding_159/embeddings/vAdam/embedding_160/embeddings/v*^
TinW
U2S*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference__traced_restore_256362052??
?Z
?

N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360193

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_80_greaterequal_y
model_52_256360108
model_52_256360110
model_52_256360112
model_52_256360114
model_53_256360117
model_53_256360119
model_53_256360121
model_53_256360123/
+tf_clip_by_value_80_clip_by_value_minimum_y'
#tf_clip_by_value_80_clip_by_value_y
dense_234_256360135
dense_234_256360137
dense_237_256360140
dense_237_256360142
dense_235_256360145
dense_235_256360147
dense_236_256360150
dense_236_256360152
dense_238_256360155
dense_238_256360157
dense_239_256360162
dense_239_256360164
dense_240_256360168
dense_240_256360170
dense_241_256360175
dense_241_256360177
normalize_26_256360182
normalize_26_256360184
dense_242_256360187
dense_242_256360189
identity??!dense_234/StatefulPartitionedCall?!dense_235/StatefulPartitionedCall?!dense_236/StatefulPartitionedCall?!dense_237/StatefulPartitionedCall?!dense_238/StatefulPartitionedCall?!dense_239/StatefulPartitionedCall?!dense_240/StatefulPartitionedCall?!dense_241/StatefulPartitionedCall?!dense_242/StatefulPartitionedCall? model_52/StatefulPartitionedCall? model_53/StatefulPartitionedCall?$normalize_26/StatefulPartitionedCall?
%tf.math.greater_equal_80/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_80/GreaterEqual?
 model_52/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_52_256360108model_52_256360110model_52_256360112model_52_256360114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563593572"
 model_52/StatefulPartitionedCall?
 model_53/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_53_256360117model_53_256360119model_53_256360121model_53_256360123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563595832"
 model_53/StatefulPartitionedCall?
)tf.clip_by_value_80/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_80/clip_by_value/Minimum?
!tf.clip_by_value_80/clip_by_valueMaximum-tf.clip_by_value_80/clip_by_value/Minimum:z:0#tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_80/clip_by_value?
tf.cast_80/CastCast)tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_80/Castv
tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_78/concat/axis?
tf.concat_78/concatConcatV2)model_52/StatefulPartitionedCall:output:0)model_53/StatefulPartitionedCall:output:0!tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_78/concat
tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_79/concat/axis?
tf.concat_79/concatConcatV2%tf.clip_by_value_80/clip_by_value:z:0tf.cast_80/Cast:y:0!tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_79/concat?
!dense_234/StatefulPartitionedCallStatefulPartitionedCalltf.concat_78/concat:output:0dense_234_256360135dense_234_256360137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_234_layer_call_and_return_conditional_losses_2563597372#
!dense_234/StatefulPartitionedCall?
!dense_237/StatefulPartitionedCallStatefulPartitionedCalltf.concat_79/concat:output:0dense_237_256360140dense_237_256360142*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_237_layer_call_and_return_conditional_losses_2563597632#
!dense_237/StatefulPartitionedCall?
!dense_235/StatefulPartitionedCallStatefulPartitionedCall*dense_234/StatefulPartitionedCall:output:0dense_235_256360145dense_235_256360147*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_235_layer_call_and_return_conditional_losses_2563597902#
!dense_235/StatefulPartitionedCall?
!dense_236/StatefulPartitionedCallStatefulPartitionedCall*dense_235/StatefulPartitionedCall:output:0dense_236_256360150dense_236_256360152*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_236_layer_call_and_return_conditional_losses_2563598172#
!dense_236/StatefulPartitionedCall?
!dense_238/StatefulPartitionedCallStatefulPartitionedCall*dense_237/StatefulPartitionedCall:output:0dense_238_256360155dense_238_256360157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_238_layer_call_and_return_conditional_losses_2563598432#
!dense_238/StatefulPartitionedCall
tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_80/concat/axis?
tf.concat_80/concatConcatV2*dense_236/StatefulPartitionedCall:output:0*dense_238/StatefulPartitionedCall:output:0!tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_80/concat?
!dense_239/StatefulPartitionedCallStatefulPartitionedCalltf.concat_80/concat:output:0dense_239_256360162dense_239_256360164*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_239_layer_call_and_return_conditional_losses_2563598712#
!dense_239/StatefulPartitionedCall?
tf.nn.relu_78/ReluRelu*dense_239/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_78/Relu?
!dense_240/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_78/Relu:activations:0dense_240_256360168dense_240_256360170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_240_layer_call_and_return_conditional_losses_2563598982#
!dense_240/StatefulPartitionedCall?
tf.__operators__.add_160/AddV2AddV2*dense_240/StatefulPartitionedCall:output:0 tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_160/AddV2?
tf.nn.relu_79/ReluRelu"tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_79/Relu?
!dense_241/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_79/Relu:activations:0dense_241_256360175dense_241_256360177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_241_layer_call_and_return_conditional_losses_2563599262#
!dense_241/StatefulPartitionedCall?
tf.__operators__.add_161/AddV2AddV2*dense_241/StatefulPartitionedCall:output:0 tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_161/AddV2?
tf.nn.relu_80/ReluRelu"tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_80/Relu?
$normalize_26/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_80/Relu:activations:0normalize_26_256360182normalize_26_256360184*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_normalize_26_layer_call_and_return_conditional_losses_2563599612&
$normalize_26/StatefulPartitionedCall?
!dense_242/StatefulPartitionedCallStatefulPartitionedCall-normalize_26/StatefulPartitionedCall:output:0dense_242_256360187dense_242_256360189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_242_layer_call_and_return_conditional_losses_2563599872#
!dense_242/StatefulPartitionedCall?
IdentityIdentity*dense_242/StatefulPartitionedCall:output:0"^dense_234/StatefulPartitionedCall"^dense_235/StatefulPartitionedCall"^dense_236/StatefulPartitionedCall"^dense_237/StatefulPartitionedCall"^dense_238/StatefulPartitionedCall"^dense_239/StatefulPartitionedCall"^dense_240/StatefulPartitionedCall"^dense_241/StatefulPartitionedCall"^dense_242/StatefulPartitionedCall!^model_52/StatefulPartitionedCall!^model_53/StatefulPartitionedCall%^normalize_26/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2F
!dense_234/StatefulPartitionedCall!dense_234/StatefulPartitionedCall2F
!dense_235/StatefulPartitionedCall!dense_235/StatefulPartitionedCall2F
!dense_236/StatefulPartitionedCall!dense_236/StatefulPartitionedCall2F
!dense_237/StatefulPartitionedCall!dense_237/StatefulPartitionedCall2F
!dense_238/StatefulPartitionedCall!dense_238/StatefulPartitionedCall2F
!dense_239/StatefulPartitionedCall!dense_239/StatefulPartitionedCall2F
!dense_240/StatefulPartitionedCall!dense_240/StatefulPartitionedCall2F
!dense_241/StatefulPartitionedCall!dense_241/StatefulPartitionedCall2F
!dense_242/StatefulPartitionedCall!dense_242/StatefulPartitionedCall2D
 model_52/StatefulPartitionedCall model_52/StatefulPartitionedCall2D
 model_53/StatefulPartitionedCall model_53/StatefulPartitionedCall2L
$normalize_26/StatefulPartitionedCall$normalize_26/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?.
?
G__inference_model_53_layer_call_and_return_conditional_losses_256359583

inputs+
'tf_math_greater_equal_79_greaterequal_y
embedding_161_256359565
embedding_159_256359568
embedding_160_256359573
identity??%embedding_159/StatefulPartitionedCall?%embedding_160/StatefulPartitionedCall?%embedding_161/StatefulPartitionedCall?
flatten_53/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_53_layer_call_and_return_conditional_losses_2563594232
flatten_53/PartitionedCall?
+tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_79/clip_by_value/Minimum/y?
)tf.clip_by_value_79/clip_by_value/MinimumMinimum#flatten_53/PartitionedCall:output:04tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_79/clip_by_value/Minimum?
#tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_79/clip_by_value/y?
!tf.clip_by_value_79/clip_by_valueMaximum-tf.clip_by_value_79/clip_by_value/Minimum:z:0,tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_79/clip_by_value?
$tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_53/FloorDiv/y?
"tf.compat.v1.floor_div_53/FloorDivFloorDiv%tf.clip_by_value_79/clip_by_value:z:0-tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_53/FloorDiv?
%tf.math.greater_equal_79/GreaterEqualGreaterEqual#flatten_53/PartitionedCall:output:0'tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_79/GreaterEqual?
tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_53/FloorMod/y?
tf.math.floormod_53/FloorModFloorMod%tf.clip_by_value_79/clip_by_value:z:0'tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_53/FloorMod?
%embedding_161/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_79/clip_by_value:z:0embedding_161_256359565*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_161_layer_call_and_return_conditional_losses_2563594512'
%embedding_161/StatefulPartitionedCall?
%embedding_159/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_53/FloorDiv:z:0embedding_159_256359568*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_159_layer_call_and_return_conditional_losses_2563594732'
%embedding_159/StatefulPartitionedCall?
tf.cast_79/CastCast)tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_79/Cast?
tf.__operators__.add_158/AddV2AddV2.embedding_161/StatefulPartitionedCall:output:0.embedding_159/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_158/AddV2?
%embedding_160/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_53/FloorMod:z:0embedding_160_256359573*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_160_layer_call_and_return_conditional_losses_2563594972'
%embedding_160/StatefulPartitionedCall?
tf.__operators__.add_159/AddV2AddV2"tf.__operators__.add_158/AddV2:z:0.embedding_160/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_159/AddV2?
 tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_53/ExpandDims/dim?
tf.expand_dims_53/ExpandDims
ExpandDimstf.cast_79/Cast:y:0)tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_53/ExpandDims?
tf.math.multiply_53/MulMul"tf.__operators__.add_159/AddV2:z:0%tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_53/Mul?
+tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_53/Sum/reduction_indices?
tf.math.reduce_sum_53/SumSumtf.math.multiply_53/Mul:z:04tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_53/Sum?
IdentityIdentity"tf.math.reduce_sum_53/Sum:output:0&^embedding_159/StatefulPartitionedCall&^embedding_160/StatefulPartitionedCall&^embedding_161/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_159/StatefulPartitionedCall%embedding_159/StatefulPartitionedCall2N
%embedding_160/StatefulPartitionedCall%embedding_160/StatefulPartitionedCall2N
%embedding_161/StatefulPartitionedCall%embedding_161/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
3__inference_custom_model_26_layer_call_fn_256360419

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_custom_model_26_layer_call_and_return_conditional_losses_2563603542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_model_52_layer_call_fn_256361073

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563593572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?:
?
G__inference_model_53_layer_call_and_return_conditional_losses_256361128

inputs+
'tf_math_greater_equal_79_greaterequal_y,
(embedding_161_embedding_lookup_256361102,
(embedding_159_embedding_lookup_256361108,
(embedding_160_embedding_lookup_256361116
identity??embedding_159/embedding_lookup?embedding_160/embedding_lookup?embedding_161/embedding_lookupu
flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_53/Const?
flatten_53/ReshapeReshapeinputsflatten_53/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_53/Reshape?
+tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_79/clip_by_value/Minimum/y?
)tf.clip_by_value_79/clip_by_value/MinimumMinimumflatten_53/Reshape:output:04tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_79/clip_by_value/Minimum?
#tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_79/clip_by_value/y?
!tf.clip_by_value_79/clip_by_valueMaximum-tf.clip_by_value_79/clip_by_value/Minimum:z:0,tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_79/clip_by_value?
$tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_53/FloorDiv/y?
"tf.compat.v1.floor_div_53/FloorDivFloorDiv%tf.clip_by_value_79/clip_by_value:z:0-tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_53/FloorDiv?
%tf.math.greater_equal_79/GreaterEqualGreaterEqualflatten_53/Reshape:output:0'tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_79/GreaterEqual?
tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_53/FloorMod/y?
tf.math.floormod_53/FloorModFloorMod%tf.clip_by_value_79/clip_by_value:z:0'tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_53/FloorMod?
embedding_161/CastCast%tf.clip_by_value_79/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_161/Cast?
embedding_161/embedding_lookupResourceGather(embedding_161_embedding_lookup_256361102embedding_161/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_161/embedding_lookup/256361102*,
_output_shapes
:??????????*
dtype02 
embedding_161/embedding_lookup?
'embedding_161/embedding_lookup/IdentityIdentity'embedding_161/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_161/embedding_lookup/256361102*,
_output_shapes
:??????????2)
'embedding_161/embedding_lookup/Identity?
)embedding_161/embedding_lookup/Identity_1Identity0embedding_161/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_161/embedding_lookup/Identity_1?
embedding_159/CastCast&tf.compat.v1.floor_div_53/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_159/Cast?
embedding_159/embedding_lookupResourceGather(embedding_159_embedding_lookup_256361108embedding_159/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_159/embedding_lookup/256361108*,
_output_shapes
:??????????*
dtype02 
embedding_159/embedding_lookup?
'embedding_159/embedding_lookup/IdentityIdentity'embedding_159/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_159/embedding_lookup/256361108*,
_output_shapes
:??????????2)
'embedding_159/embedding_lookup/Identity?
)embedding_159/embedding_lookup/Identity_1Identity0embedding_159/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_159/embedding_lookup/Identity_1?
tf.cast_79/CastCast)tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_79/Cast?
tf.__operators__.add_158/AddV2AddV22embedding_161/embedding_lookup/Identity_1:output:02embedding_159/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_158/AddV2?
embedding_160/CastCast tf.math.floormod_53/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_160/Cast?
embedding_160/embedding_lookupResourceGather(embedding_160_embedding_lookup_256361116embedding_160/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_160/embedding_lookup/256361116*,
_output_shapes
:??????????*
dtype02 
embedding_160/embedding_lookup?
'embedding_160/embedding_lookup/IdentityIdentity'embedding_160/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_160/embedding_lookup/256361116*,
_output_shapes
:??????????2)
'embedding_160/embedding_lookup/Identity?
)embedding_160/embedding_lookup/Identity_1Identity0embedding_160/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_160/embedding_lookup/Identity_1?
tf.__operators__.add_159/AddV2AddV2"tf.__operators__.add_158/AddV2:z:02embedding_160/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_159/AddV2?
 tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_53/ExpandDims/dim?
tf.expand_dims_53/ExpandDims
ExpandDimstf.cast_79/Cast:y:0)tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_53/ExpandDims?
tf.math.multiply_53/MulMul"tf.__operators__.add_159/AddV2:z:0%tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_53/Mul?
+tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_53/Sum/reduction_indices?
tf.math.reduce_sum_53/SumSumtf.math.multiply_53/Mul:z:04tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_53/Sum?
IdentityIdentity"tf.math.reduce_sum_53/Sum:output:0^embedding_159/embedding_lookup^embedding_160/embedding_lookup^embedding_161/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2@
embedding_159/embedding_lookupembedding_159/embedding_lookup2@
embedding_160/embedding_lookupembedding_160/embedding_lookup2@
embedding_161/embedding_lookupembedding_161/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
H__inference_dense_240_layer_call_and_return_conditional_losses_256359898

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
w
1__inference_embedding_156_layer_call_fn_256361441

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_156_layer_call_and_return_conditional_losses_2563592472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_238_layer_call_and_return_conditional_losses_256359843

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
G__inference_model_52_layer_call_and_return_conditional_losses_256359357

inputs+
'tf_math_greater_equal_78_greaterequal_y
embedding_158_256359339
embedding_156_256359342
embedding_157_256359347
identity??%embedding_156/StatefulPartitionedCall?%embedding_157/StatefulPartitionedCall?%embedding_158/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_52_layer_call_and_return_conditional_losses_2563591972
flatten_52/PartitionedCall?
+tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_78/clip_by_value/Minimum/y?
)tf.clip_by_value_78/clip_by_value/MinimumMinimum#flatten_52/PartitionedCall:output:04tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_78/clip_by_value/Minimum?
#tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_78/clip_by_value/y?
!tf.clip_by_value_78/clip_by_valueMaximum-tf.clip_by_value_78/clip_by_value/Minimum:z:0,tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_78/clip_by_value?
$tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_52/FloorDiv/y?
"tf.compat.v1.floor_div_52/FloorDivFloorDiv%tf.clip_by_value_78/clip_by_value:z:0-tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_52/FloorDiv?
%tf.math.greater_equal_78/GreaterEqualGreaterEqual#flatten_52/PartitionedCall:output:0'tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_78/GreaterEqual?
tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_52/FloorMod/y?
tf.math.floormod_52/FloorModFloorMod%tf.clip_by_value_78/clip_by_value:z:0'tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_52/FloorMod?
%embedding_158/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_78/clip_by_value:z:0embedding_158_256359339*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_158_layer_call_and_return_conditional_losses_2563592252'
%embedding_158/StatefulPartitionedCall?
%embedding_156/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_52/FloorDiv:z:0embedding_156_256359342*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_156_layer_call_and_return_conditional_losses_2563592472'
%embedding_156/StatefulPartitionedCall?
tf.cast_78/CastCast)tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_78/Cast?
tf.__operators__.add_156/AddV2AddV2.embedding_158/StatefulPartitionedCall:output:0.embedding_156/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_156/AddV2?
%embedding_157/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_52/FloorMod:z:0embedding_157_256359347*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_157_layer_call_and_return_conditional_losses_2563592712'
%embedding_157/StatefulPartitionedCall?
tf.__operators__.add_157/AddV2AddV2"tf.__operators__.add_156/AddV2:z:0.embedding_157/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_157/AddV2?
 tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_52/ExpandDims/dim?
tf.expand_dims_52/ExpandDims
ExpandDimstf.cast_78/Cast:y:0)tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_52/ExpandDims?
tf.math.multiply_52/MulMul"tf.__operators__.add_157/AddV2:z:0%tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_52/Mul?
+tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_52/Sum/reduction_indices?
tf.math.reduce_sum_52/SumSumtf.math.multiply_52/Mul:z:04tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_52/Sum?
IdentityIdentity"tf.math.reduce_sum_52/Sum:output:0&^embedding_156/StatefulPartitionedCall&^embedding_157/StatefulPartitionedCall&^embedding_158/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_156/StatefulPartitionedCall%embedding_156/StatefulPartitionedCall2N
%embedding_157/StatefulPartitionedCall%embedding_157/StatefulPartitionedCall2N
%embedding_158/StatefulPartitionedCall%embedding_158/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
-__inference_dense_240_layer_call_fn_256361332

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_240_layer_call_and_return_conditional_losses_2563598982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
G__inference_model_52_layer_call_and_return_conditional_losses_256359402

inputs+
'tf_math_greater_equal_78_greaterequal_y
embedding_158_256359384
embedding_156_256359387
embedding_157_256359392
identity??%embedding_156/StatefulPartitionedCall?%embedding_157/StatefulPartitionedCall?%embedding_158/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_52_layer_call_and_return_conditional_losses_2563591972
flatten_52/PartitionedCall?
+tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_78/clip_by_value/Minimum/y?
)tf.clip_by_value_78/clip_by_value/MinimumMinimum#flatten_52/PartitionedCall:output:04tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_78/clip_by_value/Minimum?
#tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_78/clip_by_value/y?
!tf.clip_by_value_78/clip_by_valueMaximum-tf.clip_by_value_78/clip_by_value/Minimum:z:0,tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_78/clip_by_value?
$tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_52/FloorDiv/y?
"tf.compat.v1.floor_div_52/FloorDivFloorDiv%tf.clip_by_value_78/clip_by_value:z:0-tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_52/FloorDiv?
%tf.math.greater_equal_78/GreaterEqualGreaterEqual#flatten_52/PartitionedCall:output:0'tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_78/GreaterEqual?
tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_52/FloorMod/y?
tf.math.floormod_52/FloorModFloorMod%tf.clip_by_value_78/clip_by_value:z:0'tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_52/FloorMod?
%embedding_158/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_78/clip_by_value:z:0embedding_158_256359384*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_158_layer_call_and_return_conditional_losses_2563592252'
%embedding_158/StatefulPartitionedCall?
%embedding_156/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_52/FloorDiv:z:0embedding_156_256359387*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_156_layer_call_and_return_conditional_losses_2563592472'
%embedding_156/StatefulPartitionedCall?
tf.cast_78/CastCast)tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_78/Cast?
tf.__operators__.add_156/AddV2AddV2.embedding_158/StatefulPartitionedCall:output:0.embedding_156/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_156/AddV2?
%embedding_157/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_52/FloorMod:z:0embedding_157_256359392*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_157_layer_call_and_return_conditional_losses_2563592712'
%embedding_157/StatefulPartitionedCall?
tf.__operators__.add_157/AddV2AddV2"tf.__operators__.add_156/AddV2:z:0.embedding_157/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_157/AddV2?
 tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_52/ExpandDims/dim?
tf.expand_dims_52/ExpandDims
ExpandDimstf.cast_78/Cast:y:0)tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_52/ExpandDims?
tf.math.multiply_52/MulMul"tf.__operators__.add_157/AddV2:z:0%tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_52/Mul?
+tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_52/Sum/reduction_indices?
tf.math.reduce_sum_52/SumSumtf.math.multiply_52/Mul:z:04tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_52/Sum?
IdentityIdentity"tf.math.reduce_sum_52/Sum:output:0&^embedding_156/StatefulPartitionedCall&^embedding_157/StatefulPartitionedCall&^embedding_158/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_156/StatefulPartitionedCall%embedding_156/StatefulPartitionedCall2N
%embedding_157/StatefulPartitionedCall%embedding_157/StatefulPartitionedCall2N
%embedding_158/StatefulPartitionedCall%embedding_158/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360668

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_80_greaterequal_y4
0model_52_tf_math_greater_equal_78_greaterequal_y5
1model_52_embedding_158_embedding_lookup_2563605185
1model_52_embedding_156_embedding_lookup_2563605245
1model_52_embedding_157_embedding_lookup_2563605324
0model_53_tf_math_greater_equal_79_greaterequal_y5
1model_53_embedding_161_embedding_lookup_2563605565
1model_53_embedding_159_embedding_lookup_2563605625
1model_53_embedding_160_embedding_lookup_256360570/
+tf_clip_by_value_80_clip_by_value_minimum_y'
#tf_clip_by_value_80_clip_by_value_y,
(dense_234_matmul_readvariableop_resource-
)dense_234_biasadd_readvariableop_resource,
(dense_237_matmul_readvariableop_resource-
)dense_237_biasadd_readvariableop_resource,
(dense_235_matmul_readvariableop_resource-
)dense_235_biasadd_readvariableop_resource,
(dense_236_matmul_readvariableop_resource-
)dense_236_biasadd_readvariableop_resource,
(dense_238_matmul_readvariableop_resource-
)dense_238_biasadd_readvariableop_resource,
(dense_239_matmul_readvariableop_resource-
)dense_239_biasadd_readvariableop_resource,
(dense_240_matmul_readvariableop_resource-
)dense_240_biasadd_readvariableop_resource,
(dense_241_matmul_readvariableop_resource-
)dense_241_biasadd_readvariableop_resourceA
=normalize_26_normalization_26_reshape_readvariableop_resourceC
?normalize_26_normalization_26_reshape_1_readvariableop_resource,
(dense_242_matmul_readvariableop_resource-
)dense_242_biasadd_readvariableop_resource
identity?? dense_234/BiasAdd/ReadVariableOp?dense_234/MatMul/ReadVariableOp? dense_235/BiasAdd/ReadVariableOp?dense_235/MatMul/ReadVariableOp? dense_236/BiasAdd/ReadVariableOp?dense_236/MatMul/ReadVariableOp? dense_237/BiasAdd/ReadVariableOp?dense_237/MatMul/ReadVariableOp? dense_238/BiasAdd/ReadVariableOp?dense_238/MatMul/ReadVariableOp? dense_239/BiasAdd/ReadVariableOp?dense_239/MatMul/ReadVariableOp? dense_240/BiasAdd/ReadVariableOp?dense_240/MatMul/ReadVariableOp? dense_241/BiasAdd/ReadVariableOp?dense_241/MatMul/ReadVariableOp? dense_242/BiasAdd/ReadVariableOp?dense_242/MatMul/ReadVariableOp?'model_52/embedding_156/embedding_lookup?'model_52/embedding_157/embedding_lookup?'model_52/embedding_158/embedding_lookup?'model_53/embedding_159/embedding_lookup?'model_53/embedding_160/embedding_lookup?'model_53/embedding_161/embedding_lookup?4normalize_26/normalization_26/Reshape/ReadVariableOp?6normalize_26/normalization_26/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_80/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_80/GreaterEqual?
model_52/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_52/flatten_52/Const?
model_52/flatten_52/ReshapeReshape
inputs_0_0"model_52/flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2
model_52/flatten_52/Reshape?
4model_52/tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_52/tf.clip_by_value_78/clip_by_value/Minimum/y?
2model_52/tf.clip_by_value_78/clip_by_value/MinimumMinimum$model_52/flatten_52/Reshape:output:0=model_52/tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_52/tf.clip_by_value_78/clip_by_value/Minimum?
,model_52/tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_52/tf.clip_by_value_78/clip_by_value/y?
*model_52/tf.clip_by_value_78/clip_by_valueMaximum6model_52/tf.clip_by_value_78/clip_by_value/Minimum:z:05model_52/tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_52/tf.clip_by_value_78/clip_by_value?
-model_52/tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_52/tf.compat.v1.floor_div_52/FloorDiv/y?
+model_52/tf.compat.v1.floor_div_52/FloorDivFloorDiv.model_52/tf.clip_by_value_78/clip_by_value:z:06model_52/tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_52/tf.compat.v1.floor_div_52/FloorDiv?
.model_52/tf.math.greater_equal_78/GreaterEqualGreaterEqual$model_52/flatten_52/Reshape:output:00model_52_tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_52/tf.math.greater_equal_78/GreaterEqual?
'model_52/tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_52/tf.math.floormod_52/FloorMod/y?
%model_52/tf.math.floormod_52/FloorModFloorMod.model_52/tf.clip_by_value_78/clip_by_value:z:00model_52/tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_52/tf.math.floormod_52/FloorMod?
model_52/embedding_158/CastCast.model_52/tf.clip_by_value_78/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_52/embedding_158/Cast?
'model_52/embedding_158/embedding_lookupResourceGather1model_52_embedding_158_embedding_lookup_256360518model_52/embedding_158/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_52/embedding_158/embedding_lookup/256360518*,
_output_shapes
:??????????*
dtype02)
'model_52/embedding_158/embedding_lookup?
0model_52/embedding_158/embedding_lookup/IdentityIdentity0model_52/embedding_158/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_52/embedding_158/embedding_lookup/256360518*,
_output_shapes
:??????????22
0model_52/embedding_158/embedding_lookup/Identity?
2model_52/embedding_158/embedding_lookup/Identity_1Identity9model_52/embedding_158/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_52/embedding_158/embedding_lookup/Identity_1?
model_52/embedding_156/CastCast/model_52/tf.compat.v1.floor_div_52/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_52/embedding_156/Cast?
'model_52/embedding_156/embedding_lookupResourceGather1model_52_embedding_156_embedding_lookup_256360524model_52/embedding_156/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_52/embedding_156/embedding_lookup/256360524*,
_output_shapes
:??????????*
dtype02)
'model_52/embedding_156/embedding_lookup?
0model_52/embedding_156/embedding_lookup/IdentityIdentity0model_52/embedding_156/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_52/embedding_156/embedding_lookup/256360524*,
_output_shapes
:??????????22
0model_52/embedding_156/embedding_lookup/Identity?
2model_52/embedding_156/embedding_lookup/Identity_1Identity9model_52/embedding_156/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_52/embedding_156/embedding_lookup/Identity_1?
model_52/tf.cast_78/CastCast2model_52/tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_52/tf.cast_78/Cast?
'model_52/tf.__operators__.add_156/AddV2AddV2;model_52/embedding_158/embedding_lookup/Identity_1:output:0;model_52/embedding_156/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_52/tf.__operators__.add_156/AddV2?
model_52/embedding_157/CastCast)model_52/tf.math.floormod_52/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_52/embedding_157/Cast?
'model_52/embedding_157/embedding_lookupResourceGather1model_52_embedding_157_embedding_lookup_256360532model_52/embedding_157/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_52/embedding_157/embedding_lookup/256360532*,
_output_shapes
:??????????*
dtype02)
'model_52/embedding_157/embedding_lookup?
0model_52/embedding_157/embedding_lookup/IdentityIdentity0model_52/embedding_157/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_52/embedding_157/embedding_lookup/256360532*,
_output_shapes
:??????????22
0model_52/embedding_157/embedding_lookup/Identity?
2model_52/embedding_157/embedding_lookup/Identity_1Identity9model_52/embedding_157/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_52/embedding_157/embedding_lookup/Identity_1?
'model_52/tf.__operators__.add_157/AddV2AddV2+model_52/tf.__operators__.add_156/AddV2:z:0;model_52/embedding_157/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_52/tf.__operators__.add_157/AddV2?
)model_52/tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_52/tf.expand_dims_52/ExpandDims/dim?
%model_52/tf.expand_dims_52/ExpandDims
ExpandDimsmodel_52/tf.cast_78/Cast:y:02model_52/tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_52/tf.expand_dims_52/ExpandDims?
 model_52/tf.math.multiply_52/MulMul+model_52/tf.__operators__.add_157/AddV2:z:0.model_52/tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_52/tf.math.multiply_52/Mul?
4model_52/tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_52/tf.math.reduce_sum_52/Sum/reduction_indices?
"model_52/tf.math.reduce_sum_52/SumSum$model_52/tf.math.multiply_52/Mul:z:0=model_52/tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_52/tf.math.reduce_sum_52/Sum?
model_53/flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_53/flatten_53/Const?
model_53/flatten_53/ReshapeReshape
inputs_0_1"model_53/flatten_53/Const:output:0*
T0*'
_output_shapes
:?????????2
model_53/flatten_53/Reshape?
4model_53/tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_53/tf.clip_by_value_79/clip_by_value/Minimum/y?
2model_53/tf.clip_by_value_79/clip_by_value/MinimumMinimum$model_53/flatten_53/Reshape:output:0=model_53/tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_53/tf.clip_by_value_79/clip_by_value/Minimum?
,model_53/tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_53/tf.clip_by_value_79/clip_by_value/y?
*model_53/tf.clip_by_value_79/clip_by_valueMaximum6model_53/tf.clip_by_value_79/clip_by_value/Minimum:z:05model_53/tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_53/tf.clip_by_value_79/clip_by_value?
-model_53/tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_53/tf.compat.v1.floor_div_53/FloorDiv/y?
+model_53/tf.compat.v1.floor_div_53/FloorDivFloorDiv.model_53/tf.clip_by_value_79/clip_by_value:z:06model_53/tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_53/tf.compat.v1.floor_div_53/FloorDiv?
.model_53/tf.math.greater_equal_79/GreaterEqualGreaterEqual$model_53/flatten_53/Reshape:output:00model_53_tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_53/tf.math.greater_equal_79/GreaterEqual?
'model_53/tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_53/tf.math.floormod_53/FloorMod/y?
%model_53/tf.math.floormod_53/FloorModFloorMod.model_53/tf.clip_by_value_79/clip_by_value:z:00model_53/tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_53/tf.math.floormod_53/FloorMod?
model_53/embedding_161/CastCast.model_53/tf.clip_by_value_79/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_53/embedding_161/Cast?
'model_53/embedding_161/embedding_lookupResourceGather1model_53_embedding_161_embedding_lookup_256360556model_53/embedding_161/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_53/embedding_161/embedding_lookup/256360556*,
_output_shapes
:??????????*
dtype02)
'model_53/embedding_161/embedding_lookup?
0model_53/embedding_161/embedding_lookup/IdentityIdentity0model_53/embedding_161/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_53/embedding_161/embedding_lookup/256360556*,
_output_shapes
:??????????22
0model_53/embedding_161/embedding_lookup/Identity?
2model_53/embedding_161/embedding_lookup/Identity_1Identity9model_53/embedding_161/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_53/embedding_161/embedding_lookup/Identity_1?
model_53/embedding_159/CastCast/model_53/tf.compat.v1.floor_div_53/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_53/embedding_159/Cast?
'model_53/embedding_159/embedding_lookupResourceGather1model_53_embedding_159_embedding_lookup_256360562model_53/embedding_159/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_53/embedding_159/embedding_lookup/256360562*,
_output_shapes
:??????????*
dtype02)
'model_53/embedding_159/embedding_lookup?
0model_53/embedding_159/embedding_lookup/IdentityIdentity0model_53/embedding_159/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_53/embedding_159/embedding_lookup/256360562*,
_output_shapes
:??????????22
0model_53/embedding_159/embedding_lookup/Identity?
2model_53/embedding_159/embedding_lookup/Identity_1Identity9model_53/embedding_159/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_53/embedding_159/embedding_lookup/Identity_1?
model_53/tf.cast_79/CastCast2model_53/tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_53/tf.cast_79/Cast?
'model_53/tf.__operators__.add_158/AddV2AddV2;model_53/embedding_161/embedding_lookup/Identity_1:output:0;model_53/embedding_159/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_53/tf.__operators__.add_158/AddV2?
model_53/embedding_160/CastCast)model_53/tf.math.floormod_53/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_53/embedding_160/Cast?
'model_53/embedding_160/embedding_lookupResourceGather1model_53_embedding_160_embedding_lookup_256360570model_53/embedding_160/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_53/embedding_160/embedding_lookup/256360570*,
_output_shapes
:??????????*
dtype02)
'model_53/embedding_160/embedding_lookup?
0model_53/embedding_160/embedding_lookup/IdentityIdentity0model_53/embedding_160/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_53/embedding_160/embedding_lookup/256360570*,
_output_shapes
:??????????22
0model_53/embedding_160/embedding_lookup/Identity?
2model_53/embedding_160/embedding_lookup/Identity_1Identity9model_53/embedding_160/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_53/embedding_160/embedding_lookup/Identity_1?
'model_53/tf.__operators__.add_159/AddV2AddV2+model_53/tf.__operators__.add_158/AddV2:z:0;model_53/embedding_160/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_53/tf.__operators__.add_159/AddV2?
)model_53/tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_53/tf.expand_dims_53/ExpandDims/dim?
%model_53/tf.expand_dims_53/ExpandDims
ExpandDimsmodel_53/tf.cast_79/Cast:y:02model_53/tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_53/tf.expand_dims_53/ExpandDims?
 model_53/tf.math.multiply_53/MulMul+model_53/tf.__operators__.add_159/AddV2:z:0.model_53/tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_53/tf.math.multiply_53/Mul?
4model_53/tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_53/tf.math.reduce_sum_53/Sum/reduction_indices?
"model_53/tf.math.reduce_sum_53/SumSum$model_53/tf.math.multiply_53/Mul:z:0=model_53/tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_53/tf.math.reduce_sum_53/Sum?
)tf.clip_by_value_80/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_80/clip_by_value/Minimum?
!tf.clip_by_value_80/clip_by_valueMaximum-tf.clip_by_value_80/clip_by_value/Minimum:z:0#tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_80/clip_by_value?
tf.cast_80/CastCast)tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_80/Castv
tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_78/concat/axis?
tf.concat_78/concatConcatV2+model_52/tf.math.reduce_sum_52/Sum:output:0+model_53/tf.math.reduce_sum_53/Sum:output:0!tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_78/concat
tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_79/concat/axis?
tf.concat_79/concatConcatV2%tf.clip_by_value_80/clip_by_value:z:0tf.cast_80/Cast:y:0!tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_79/concat?
dense_234/MatMul/ReadVariableOpReadVariableOp(dense_234_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_234/MatMul/ReadVariableOp?
dense_234/MatMulMatMultf.concat_78/concat:output:0'dense_234/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_234/MatMul?
 dense_234/BiasAdd/ReadVariableOpReadVariableOp)dense_234_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_234/BiasAdd/ReadVariableOp?
dense_234/BiasAddBiasAdddense_234/MatMul:product:0(dense_234/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_234/BiasAddw
dense_234/ReluReludense_234/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_234/Relu?
dense_237/MatMul/ReadVariableOpReadVariableOp(dense_237_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_237/MatMul/ReadVariableOp?
dense_237/MatMulMatMultf.concat_79/concat:output:0'dense_237/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_237/MatMul?
 dense_237/BiasAdd/ReadVariableOpReadVariableOp)dense_237_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_237/BiasAdd/ReadVariableOp?
dense_237/BiasAddBiasAdddense_237/MatMul:product:0(dense_237/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_237/BiasAdd?
dense_235/MatMul/ReadVariableOpReadVariableOp(dense_235_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_235/MatMul/ReadVariableOp?
dense_235/MatMulMatMuldense_234/Relu:activations:0'dense_235/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_235/MatMul?
 dense_235/BiasAdd/ReadVariableOpReadVariableOp)dense_235_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_235/BiasAdd/ReadVariableOp?
dense_235/BiasAddBiasAdddense_235/MatMul:product:0(dense_235/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_235/BiasAddw
dense_235/ReluReludense_235/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_235/Relu?
dense_236/MatMul/ReadVariableOpReadVariableOp(dense_236_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_236/MatMul/ReadVariableOp?
dense_236/MatMulMatMuldense_235/Relu:activations:0'dense_236/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_236/MatMul?
 dense_236/BiasAdd/ReadVariableOpReadVariableOp)dense_236_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_236/BiasAdd/ReadVariableOp?
dense_236/BiasAddBiasAdddense_236/MatMul:product:0(dense_236/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_236/BiasAddw
dense_236/ReluReludense_236/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_236/Relu?
dense_238/MatMul/ReadVariableOpReadVariableOp(dense_238_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_238/MatMul/ReadVariableOp?
dense_238/MatMulMatMuldense_237/BiasAdd:output:0'dense_238/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_238/MatMul?
 dense_238/BiasAdd/ReadVariableOpReadVariableOp)dense_238_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_238/BiasAdd/ReadVariableOp?
dense_238/BiasAddBiasAdddense_238/MatMul:product:0(dense_238/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_238/BiasAdd
tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_80/concat/axis?
tf.concat_80/concatConcatV2dense_236/Relu:activations:0dense_238/BiasAdd:output:0!tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_80/concat?
dense_239/MatMul/ReadVariableOpReadVariableOp(dense_239_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_239/MatMul/ReadVariableOp?
dense_239/MatMulMatMultf.concat_80/concat:output:0'dense_239/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_239/MatMul?
 dense_239/BiasAdd/ReadVariableOpReadVariableOp)dense_239_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_239/BiasAdd/ReadVariableOp?
dense_239/BiasAddBiasAdddense_239/MatMul:product:0(dense_239/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_239/BiasAdd
tf.nn.relu_78/ReluReludense_239/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_78/Relu?
dense_240/MatMul/ReadVariableOpReadVariableOp(dense_240_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_240/MatMul/ReadVariableOp?
dense_240/MatMulMatMul tf.nn.relu_78/Relu:activations:0'dense_240/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_240/MatMul?
 dense_240/BiasAdd/ReadVariableOpReadVariableOp)dense_240_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_240/BiasAdd/ReadVariableOp?
dense_240/BiasAddBiasAdddense_240/MatMul:product:0(dense_240/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_240/BiasAdd?
tf.__operators__.add_160/AddV2AddV2dense_240/BiasAdd:output:0 tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_160/AddV2?
tf.nn.relu_79/ReluRelu"tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_79/Relu?
dense_241/MatMul/ReadVariableOpReadVariableOp(dense_241_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_241/MatMul/ReadVariableOp?
dense_241/MatMulMatMul tf.nn.relu_79/Relu:activations:0'dense_241/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_241/MatMul?
 dense_241/BiasAdd/ReadVariableOpReadVariableOp)dense_241_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_241/BiasAdd/ReadVariableOp?
dense_241/BiasAddBiasAdddense_241/MatMul:product:0(dense_241/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_241/BiasAdd?
tf.__operators__.add_161/AddV2AddV2dense_241/BiasAdd:output:0 tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_161/AddV2?
tf.nn.relu_80/ReluRelu"tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_80/Relu?
4normalize_26/normalization_26/Reshape/ReadVariableOpReadVariableOp=normalize_26_normalization_26_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_26/normalization_26/Reshape/ReadVariableOp?
+normalize_26/normalization_26/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_26/normalization_26/Reshape/shape?
%normalize_26/normalization_26/ReshapeReshape<normalize_26/normalization_26/Reshape/ReadVariableOp:value:04normalize_26/normalization_26/Reshape/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_26/normalization_26/Reshape?
6normalize_26/normalization_26/Reshape_1/ReadVariableOpReadVariableOp?normalize_26_normalization_26_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6normalize_26/normalization_26/Reshape_1/ReadVariableOp?
-normalize_26/normalization_26/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2/
-normalize_26/normalization_26/Reshape_1/shape?
'normalize_26/normalization_26/Reshape_1Reshape>normalize_26/normalization_26/Reshape_1/ReadVariableOp:value:06normalize_26/normalization_26/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2)
'normalize_26/normalization_26/Reshape_1?
!normalize_26/normalization_26/subSub tf.nn.relu_80/Relu:activations:0.normalize_26/normalization_26/Reshape:output:0*
T0*(
_output_shapes
:??????????2#
!normalize_26/normalization_26/sub?
"normalize_26/normalization_26/SqrtSqrt0normalize_26/normalization_26/Reshape_1:output:0*
T0*
_output_shapes
:	?2$
"normalize_26/normalization_26/Sqrt?
'normalize_26/normalization_26/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32)
'normalize_26/normalization_26/Maximum/y?
%normalize_26/normalization_26/MaximumMaximum&normalize_26/normalization_26/Sqrt:y:00normalize_26/normalization_26/Maximum/y:output:0*
T0*
_output_shapes
:	?2'
%normalize_26/normalization_26/Maximum?
%normalize_26/normalization_26/truedivRealDiv%normalize_26/normalization_26/sub:z:0)normalize_26/normalization_26/Maximum:z:0*
T0*(
_output_shapes
:??????????2'
%normalize_26/normalization_26/truediv?
dense_242/MatMul/ReadVariableOpReadVariableOp(dense_242_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_242/MatMul/ReadVariableOp?
dense_242/MatMulMatMul)normalize_26/normalization_26/truediv:z:0'dense_242/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_242/MatMul?
 dense_242/BiasAdd/ReadVariableOpReadVariableOp)dense_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_242/BiasAdd/ReadVariableOp?
dense_242/BiasAddBiasAdddense_242/MatMul:product:0(dense_242/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_242/BiasAdd?
IdentityIdentitydense_242/BiasAdd:output:0!^dense_234/BiasAdd/ReadVariableOp ^dense_234/MatMul/ReadVariableOp!^dense_235/BiasAdd/ReadVariableOp ^dense_235/MatMul/ReadVariableOp!^dense_236/BiasAdd/ReadVariableOp ^dense_236/MatMul/ReadVariableOp!^dense_237/BiasAdd/ReadVariableOp ^dense_237/MatMul/ReadVariableOp!^dense_238/BiasAdd/ReadVariableOp ^dense_238/MatMul/ReadVariableOp!^dense_239/BiasAdd/ReadVariableOp ^dense_239/MatMul/ReadVariableOp!^dense_240/BiasAdd/ReadVariableOp ^dense_240/MatMul/ReadVariableOp!^dense_241/BiasAdd/ReadVariableOp ^dense_241/MatMul/ReadVariableOp!^dense_242/BiasAdd/ReadVariableOp ^dense_242/MatMul/ReadVariableOp(^model_52/embedding_156/embedding_lookup(^model_52/embedding_157/embedding_lookup(^model_52/embedding_158/embedding_lookup(^model_53/embedding_159/embedding_lookup(^model_53/embedding_160/embedding_lookup(^model_53/embedding_161/embedding_lookup5^normalize_26/normalization_26/Reshape/ReadVariableOp7^normalize_26/normalization_26/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_234/BiasAdd/ReadVariableOp dense_234/BiasAdd/ReadVariableOp2B
dense_234/MatMul/ReadVariableOpdense_234/MatMul/ReadVariableOp2D
 dense_235/BiasAdd/ReadVariableOp dense_235/BiasAdd/ReadVariableOp2B
dense_235/MatMul/ReadVariableOpdense_235/MatMul/ReadVariableOp2D
 dense_236/BiasAdd/ReadVariableOp dense_236/BiasAdd/ReadVariableOp2B
dense_236/MatMul/ReadVariableOpdense_236/MatMul/ReadVariableOp2D
 dense_237/BiasAdd/ReadVariableOp dense_237/BiasAdd/ReadVariableOp2B
dense_237/MatMul/ReadVariableOpdense_237/MatMul/ReadVariableOp2D
 dense_238/BiasAdd/ReadVariableOp dense_238/BiasAdd/ReadVariableOp2B
dense_238/MatMul/ReadVariableOpdense_238/MatMul/ReadVariableOp2D
 dense_239/BiasAdd/ReadVariableOp dense_239/BiasAdd/ReadVariableOp2B
dense_239/MatMul/ReadVariableOpdense_239/MatMul/ReadVariableOp2D
 dense_240/BiasAdd/ReadVariableOp dense_240/BiasAdd/ReadVariableOp2B
dense_240/MatMul/ReadVariableOpdense_240/MatMul/ReadVariableOp2D
 dense_241/BiasAdd/ReadVariableOp dense_241/BiasAdd/ReadVariableOp2B
dense_241/MatMul/ReadVariableOpdense_241/MatMul/ReadVariableOp2D
 dense_242/BiasAdd/ReadVariableOp dense_242/BiasAdd/ReadVariableOp2B
dense_242/MatMul/ReadVariableOpdense_242/MatMul/ReadVariableOp2R
'model_52/embedding_156/embedding_lookup'model_52/embedding_156/embedding_lookup2R
'model_52/embedding_157/embedding_lookup'model_52/embedding_157/embedding_lookup2R
'model_52/embedding_158/embedding_lookup'model_52/embedding_158/embedding_lookup2R
'model_53/embedding_159/embedding_lookup'model_53/embedding_159/embedding_lookup2R
'model_53/embedding_160/embedding_lookup'model_53/embedding_160/embedding_lookup2R
'model_53/embedding_161/embedding_lookup'model_53/embedding_161/embedding_lookup2l
4normalize_26/normalization_26/Reshape/ReadVariableOp4normalize_26/normalization_26/Reshape/ReadVariableOp2p
6normalize_26/normalization_26/Reshape_1/ReadVariableOp6normalize_26/normalization_26/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
w
1__inference_embedding_160_layer_call_fn_256361520

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_160_layer_call_and_return_conditional_losses_2563594972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_custom_model_26_layer_call_fn_256360258

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_custom_model_26_layer_call_and_return_conditional_losses_2563601932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?Z
?

N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360096

cards0

cards1
bets+
'tf_math_greater_equal_80_greaterequal_y
model_52_256360011
model_52_256360013
model_52_256360015
model_52_256360017
model_53_256360020
model_53_256360022
model_53_256360024
model_53_256360026/
+tf_clip_by_value_80_clip_by_value_minimum_y'
#tf_clip_by_value_80_clip_by_value_y
dense_234_256360038
dense_234_256360040
dense_237_256360043
dense_237_256360045
dense_235_256360048
dense_235_256360050
dense_236_256360053
dense_236_256360055
dense_238_256360058
dense_238_256360060
dense_239_256360065
dense_239_256360067
dense_240_256360071
dense_240_256360073
dense_241_256360078
dense_241_256360080
normalize_26_256360085
normalize_26_256360087
dense_242_256360090
dense_242_256360092
identity??!dense_234/StatefulPartitionedCall?!dense_235/StatefulPartitionedCall?!dense_236/StatefulPartitionedCall?!dense_237/StatefulPartitionedCall?!dense_238/StatefulPartitionedCall?!dense_239/StatefulPartitionedCall?!dense_240/StatefulPartitionedCall?!dense_241/StatefulPartitionedCall?!dense_242/StatefulPartitionedCall? model_52/StatefulPartitionedCall? model_53/StatefulPartitionedCall?$normalize_26/StatefulPartitionedCall?
%tf.math.greater_equal_80/GreaterEqualGreaterEqualbets'tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_80/GreaterEqual?
 model_52/StatefulPartitionedCallStatefulPartitionedCallcards0model_52_256360011model_52_256360013model_52_256360015model_52_256360017*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563594022"
 model_52/StatefulPartitionedCall?
 model_53/StatefulPartitionedCallStatefulPartitionedCallcards1model_53_256360020model_53_256360022model_53_256360024model_53_256360026*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563596282"
 model_53/StatefulPartitionedCall?
)tf.clip_by_value_80/clip_by_value/MinimumMinimumbets+tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_80/clip_by_value/Minimum?
!tf.clip_by_value_80/clip_by_valueMaximum-tf.clip_by_value_80/clip_by_value/Minimum:z:0#tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_80/clip_by_value?
tf.cast_80/CastCast)tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_80/Castv
tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_78/concat/axis?
tf.concat_78/concatConcatV2)model_52/StatefulPartitionedCall:output:0)model_53/StatefulPartitionedCall:output:0!tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_78/concat
tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_79/concat/axis?
tf.concat_79/concatConcatV2%tf.clip_by_value_80/clip_by_value:z:0tf.cast_80/Cast:y:0!tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_79/concat?
!dense_234/StatefulPartitionedCallStatefulPartitionedCalltf.concat_78/concat:output:0dense_234_256360038dense_234_256360040*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_234_layer_call_and_return_conditional_losses_2563597372#
!dense_234/StatefulPartitionedCall?
!dense_237/StatefulPartitionedCallStatefulPartitionedCalltf.concat_79/concat:output:0dense_237_256360043dense_237_256360045*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_237_layer_call_and_return_conditional_losses_2563597632#
!dense_237/StatefulPartitionedCall?
!dense_235/StatefulPartitionedCallStatefulPartitionedCall*dense_234/StatefulPartitionedCall:output:0dense_235_256360048dense_235_256360050*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_235_layer_call_and_return_conditional_losses_2563597902#
!dense_235/StatefulPartitionedCall?
!dense_236/StatefulPartitionedCallStatefulPartitionedCall*dense_235/StatefulPartitionedCall:output:0dense_236_256360053dense_236_256360055*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_236_layer_call_and_return_conditional_losses_2563598172#
!dense_236/StatefulPartitionedCall?
!dense_238/StatefulPartitionedCallStatefulPartitionedCall*dense_237/StatefulPartitionedCall:output:0dense_238_256360058dense_238_256360060*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_238_layer_call_and_return_conditional_losses_2563598432#
!dense_238/StatefulPartitionedCall
tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_80/concat/axis?
tf.concat_80/concatConcatV2*dense_236/StatefulPartitionedCall:output:0*dense_238/StatefulPartitionedCall:output:0!tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_80/concat?
!dense_239/StatefulPartitionedCallStatefulPartitionedCalltf.concat_80/concat:output:0dense_239_256360065dense_239_256360067*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_239_layer_call_and_return_conditional_losses_2563598712#
!dense_239/StatefulPartitionedCall?
tf.nn.relu_78/ReluRelu*dense_239/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_78/Relu?
!dense_240/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_78/Relu:activations:0dense_240_256360071dense_240_256360073*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_240_layer_call_and_return_conditional_losses_2563598982#
!dense_240/StatefulPartitionedCall?
tf.__operators__.add_160/AddV2AddV2*dense_240/StatefulPartitionedCall:output:0 tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_160/AddV2?
tf.nn.relu_79/ReluRelu"tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_79/Relu?
!dense_241/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_79/Relu:activations:0dense_241_256360078dense_241_256360080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_241_layer_call_and_return_conditional_losses_2563599262#
!dense_241/StatefulPartitionedCall?
tf.__operators__.add_161/AddV2AddV2*dense_241/StatefulPartitionedCall:output:0 tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_161/AddV2?
tf.nn.relu_80/ReluRelu"tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_80/Relu?
$normalize_26/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_80/Relu:activations:0normalize_26_256360085normalize_26_256360087*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_normalize_26_layer_call_and_return_conditional_losses_2563599612&
$normalize_26/StatefulPartitionedCall?
!dense_242/StatefulPartitionedCallStatefulPartitionedCall-normalize_26/StatefulPartitionedCall:output:0dense_242_256360090dense_242_256360092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_242_layer_call_and_return_conditional_losses_2563599872#
!dense_242/StatefulPartitionedCall?
IdentityIdentity*dense_242/StatefulPartitionedCall:output:0"^dense_234/StatefulPartitionedCall"^dense_235/StatefulPartitionedCall"^dense_236/StatefulPartitionedCall"^dense_237/StatefulPartitionedCall"^dense_238/StatefulPartitionedCall"^dense_239/StatefulPartitionedCall"^dense_240/StatefulPartitionedCall"^dense_241/StatefulPartitionedCall"^dense_242/StatefulPartitionedCall!^model_52/StatefulPartitionedCall!^model_53/StatefulPartitionedCall%^normalize_26/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2F
!dense_234/StatefulPartitionedCall!dense_234/StatefulPartitionedCall2F
!dense_235/StatefulPartitionedCall!dense_235/StatefulPartitionedCall2F
!dense_236/StatefulPartitionedCall!dense_236/StatefulPartitionedCall2F
!dense_237/StatefulPartitionedCall!dense_237/StatefulPartitionedCall2F
!dense_238/StatefulPartitionedCall!dense_238/StatefulPartitionedCall2F
!dense_239/StatefulPartitionedCall!dense_239/StatefulPartitionedCall2F
!dense_240/StatefulPartitionedCall!dense_240/StatefulPartitionedCall2F
!dense_241/StatefulPartitionedCall!dense_241/StatefulPartitionedCall2F
!dense_242/StatefulPartitionedCall!dense_242/StatefulPartitionedCall2D
 model_52/StatefulPartitionedCall model_52/StatefulPartitionedCall2D
 model_53/StatefulPartitionedCall model_53/StatefulPartitionedCall2L
$normalize_26/StatefulPartitionedCall$normalize_26/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?Z
?

N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360004

cards0

cards1
bets+
'tf_math_greater_equal_80_greaterequal_y
model_52_256359673
model_52_256359675
model_52_256359677
model_52_256359679
model_53_256359708
model_53_256359710
model_53_256359712
model_53_256359714/
+tf_clip_by_value_80_clip_by_value_minimum_y'
#tf_clip_by_value_80_clip_by_value_y
dense_234_256359748
dense_234_256359750
dense_237_256359774
dense_237_256359776
dense_235_256359801
dense_235_256359803
dense_236_256359828
dense_236_256359830
dense_238_256359854
dense_238_256359856
dense_239_256359882
dense_239_256359884
dense_240_256359909
dense_240_256359911
dense_241_256359937
dense_241_256359939
normalize_26_256359972
normalize_26_256359974
dense_242_256359998
dense_242_256360000
identity??!dense_234/StatefulPartitionedCall?!dense_235/StatefulPartitionedCall?!dense_236/StatefulPartitionedCall?!dense_237/StatefulPartitionedCall?!dense_238/StatefulPartitionedCall?!dense_239/StatefulPartitionedCall?!dense_240/StatefulPartitionedCall?!dense_241/StatefulPartitionedCall?!dense_242/StatefulPartitionedCall? model_52/StatefulPartitionedCall? model_53/StatefulPartitionedCall?$normalize_26/StatefulPartitionedCall?
%tf.math.greater_equal_80/GreaterEqualGreaterEqualbets'tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_80/GreaterEqual?
 model_52/StatefulPartitionedCallStatefulPartitionedCallcards0model_52_256359673model_52_256359675model_52_256359677model_52_256359679*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563593572"
 model_52/StatefulPartitionedCall?
 model_53/StatefulPartitionedCallStatefulPartitionedCallcards1model_53_256359708model_53_256359710model_53_256359712model_53_256359714*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563595832"
 model_53/StatefulPartitionedCall?
)tf.clip_by_value_80/clip_by_value/MinimumMinimumbets+tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_80/clip_by_value/Minimum?
!tf.clip_by_value_80/clip_by_valueMaximum-tf.clip_by_value_80/clip_by_value/Minimum:z:0#tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_80/clip_by_value?
tf.cast_80/CastCast)tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_80/Castv
tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_78/concat/axis?
tf.concat_78/concatConcatV2)model_52/StatefulPartitionedCall:output:0)model_53/StatefulPartitionedCall:output:0!tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_78/concat
tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_79/concat/axis?
tf.concat_79/concatConcatV2%tf.clip_by_value_80/clip_by_value:z:0tf.cast_80/Cast:y:0!tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_79/concat?
!dense_234/StatefulPartitionedCallStatefulPartitionedCalltf.concat_78/concat:output:0dense_234_256359748dense_234_256359750*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_234_layer_call_and_return_conditional_losses_2563597372#
!dense_234/StatefulPartitionedCall?
!dense_237/StatefulPartitionedCallStatefulPartitionedCalltf.concat_79/concat:output:0dense_237_256359774dense_237_256359776*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_237_layer_call_and_return_conditional_losses_2563597632#
!dense_237/StatefulPartitionedCall?
!dense_235/StatefulPartitionedCallStatefulPartitionedCall*dense_234/StatefulPartitionedCall:output:0dense_235_256359801dense_235_256359803*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_235_layer_call_and_return_conditional_losses_2563597902#
!dense_235/StatefulPartitionedCall?
!dense_236/StatefulPartitionedCallStatefulPartitionedCall*dense_235/StatefulPartitionedCall:output:0dense_236_256359828dense_236_256359830*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_236_layer_call_and_return_conditional_losses_2563598172#
!dense_236/StatefulPartitionedCall?
!dense_238/StatefulPartitionedCallStatefulPartitionedCall*dense_237/StatefulPartitionedCall:output:0dense_238_256359854dense_238_256359856*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_238_layer_call_and_return_conditional_losses_2563598432#
!dense_238/StatefulPartitionedCall
tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_80/concat/axis?
tf.concat_80/concatConcatV2*dense_236/StatefulPartitionedCall:output:0*dense_238/StatefulPartitionedCall:output:0!tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_80/concat?
!dense_239/StatefulPartitionedCallStatefulPartitionedCalltf.concat_80/concat:output:0dense_239_256359882dense_239_256359884*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_239_layer_call_and_return_conditional_losses_2563598712#
!dense_239/StatefulPartitionedCall?
tf.nn.relu_78/ReluRelu*dense_239/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_78/Relu?
!dense_240/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_78/Relu:activations:0dense_240_256359909dense_240_256359911*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_240_layer_call_and_return_conditional_losses_2563598982#
!dense_240/StatefulPartitionedCall?
tf.__operators__.add_160/AddV2AddV2*dense_240/StatefulPartitionedCall:output:0 tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_160/AddV2?
tf.nn.relu_79/ReluRelu"tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_79/Relu?
!dense_241/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_79/Relu:activations:0dense_241_256359937dense_241_256359939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_241_layer_call_and_return_conditional_losses_2563599262#
!dense_241/StatefulPartitionedCall?
tf.__operators__.add_161/AddV2AddV2*dense_241/StatefulPartitionedCall:output:0 tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_161/AddV2?
tf.nn.relu_80/ReluRelu"tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_80/Relu?
$normalize_26/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_80/Relu:activations:0normalize_26_256359972normalize_26_256359974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_normalize_26_layer_call_and_return_conditional_losses_2563599612&
$normalize_26/StatefulPartitionedCall?
!dense_242/StatefulPartitionedCallStatefulPartitionedCall-normalize_26/StatefulPartitionedCall:output:0dense_242_256359998dense_242_256360000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_242_layer_call_and_return_conditional_losses_2563599872#
!dense_242/StatefulPartitionedCall?
IdentityIdentity*dense_242/StatefulPartitionedCall:output:0"^dense_234/StatefulPartitionedCall"^dense_235/StatefulPartitionedCall"^dense_236/StatefulPartitionedCall"^dense_237/StatefulPartitionedCall"^dense_238/StatefulPartitionedCall"^dense_239/StatefulPartitionedCall"^dense_240/StatefulPartitionedCall"^dense_241/StatefulPartitionedCall"^dense_242/StatefulPartitionedCall!^model_52/StatefulPartitionedCall!^model_53/StatefulPartitionedCall%^normalize_26/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2F
!dense_234/StatefulPartitionedCall!dense_234/StatefulPartitionedCall2F
!dense_235/StatefulPartitionedCall!dense_235/StatefulPartitionedCall2F
!dense_236/StatefulPartitionedCall!dense_236/StatefulPartitionedCall2F
!dense_237/StatefulPartitionedCall!dense_237/StatefulPartitionedCall2F
!dense_238/StatefulPartitionedCall!dense_238/StatefulPartitionedCall2F
!dense_239/StatefulPartitionedCall!dense_239/StatefulPartitionedCall2F
!dense_240/StatefulPartitionedCall!dense_240/StatefulPartitionedCall2F
!dense_241/StatefulPartitionedCall!dense_241/StatefulPartitionedCall2F
!dense_242/StatefulPartitionedCall!dense_242/StatefulPartitionedCall2D
 model_52/StatefulPartitionedCall model_52/StatefulPartitionedCall2D
 model_53/StatefulPartitionedCall model_53/StatefulPartitionedCall2L
$normalize_26/StatefulPartitionedCall$normalize_26/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_custom_model_26_layer_call_fn_256360907

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_custom_model_26_layer_call_and_return_conditional_losses_2563601932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_dense_238_layer_call_fn_256361294

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_238_layer_call_and_return_conditional_losses_2563598432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_dense_235_layer_call_fn_256361236

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_235_layer_call_and_return_conditional_losses_2563597902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_156_layer_call_and_return_conditional_losses_256359247

inputs
embedding_lookup_256359241
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256359241Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256359241*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256359241*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_161_layer_call_and_return_conditional_losses_256361479

inputs
embedding_lookup_256361473
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256361473Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256361473*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256361473*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_normalize_26_layer_call_and_return_conditional_losses_256361368
x4
0normalization_26_reshape_readvariableop_resource6
2normalization_26_reshape_1_readvariableop_resource
identity??'normalization_26/Reshape/ReadVariableOp?)normalization_26/Reshape_1/ReadVariableOp?
'normalization_26/Reshape/ReadVariableOpReadVariableOp0normalization_26_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'normalization_26/Reshape/ReadVariableOp?
normalization_26/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_26/Reshape/shape?
normalization_26/ReshapeReshape/normalization_26/Reshape/ReadVariableOp:value:0'normalization_26/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_26/Reshape?
)normalization_26/Reshape_1/ReadVariableOpReadVariableOp2normalization_26_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)normalization_26/Reshape_1/ReadVariableOp?
 normalization_26/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_26/Reshape_1/shape?
normalization_26/Reshape_1Reshape1normalization_26/Reshape_1/ReadVariableOp:value:0)normalization_26/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_26/Reshape_1?
normalization_26/subSubx!normalization_26/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_26/sub?
normalization_26/SqrtSqrt#normalization_26/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_26/Sqrt}
normalization_26/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_26/Maximum/y?
normalization_26/MaximumMaximumnormalization_26/Sqrt:y:0#normalization_26/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_26/Maximum?
normalization_26/truedivRealDivnormalization_26/sub:z:0normalization_26/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_26/truediv?
IdentityIdentitynormalization_26/truediv:z:0(^normalization_26/Reshape/ReadVariableOp*^normalization_26/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2R
'normalization_26/Reshape/ReadVariableOp'normalization_26/Reshape/ReadVariableOp2V
)normalization_26/Reshape_1/ReadVariableOp)normalization_26/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
3__inference_custom_model_26_layer_call_fn_256360976

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *W
fRRP
N__inference_custom_model_26_layer_call_and_return_conditional_losses_2563603542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
H__inference_dense_241_layer_call_and_return_conditional_losses_256359926

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_157_layer_call_and_return_conditional_losses_256359271

inputs
embedding_lookup_256359265
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256359265Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256359265*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256359265*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_160_layer_call_and_return_conditional_losses_256359497

inputs
embedding_lookup_256359491
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256359491Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256359491*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256359491*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
G__inference_model_52_layer_call_and_return_conditional_losses_256361018

inputs+
'tf_math_greater_equal_78_greaterequal_y,
(embedding_158_embedding_lookup_256360992,
(embedding_156_embedding_lookup_256360998,
(embedding_157_embedding_lookup_256361006
identity??embedding_156/embedding_lookup?embedding_157/embedding_lookup?embedding_158/embedding_lookupu
flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_52/Const?
flatten_52/ReshapeReshapeinputsflatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_52/Reshape?
+tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_78/clip_by_value/Minimum/y?
)tf.clip_by_value_78/clip_by_value/MinimumMinimumflatten_52/Reshape:output:04tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_78/clip_by_value/Minimum?
#tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_78/clip_by_value/y?
!tf.clip_by_value_78/clip_by_valueMaximum-tf.clip_by_value_78/clip_by_value/Minimum:z:0,tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_78/clip_by_value?
$tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_52/FloorDiv/y?
"tf.compat.v1.floor_div_52/FloorDivFloorDiv%tf.clip_by_value_78/clip_by_value:z:0-tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_52/FloorDiv?
%tf.math.greater_equal_78/GreaterEqualGreaterEqualflatten_52/Reshape:output:0'tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_78/GreaterEqual?
tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_52/FloorMod/y?
tf.math.floormod_52/FloorModFloorMod%tf.clip_by_value_78/clip_by_value:z:0'tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_52/FloorMod?
embedding_158/CastCast%tf.clip_by_value_78/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_158/Cast?
embedding_158/embedding_lookupResourceGather(embedding_158_embedding_lookup_256360992embedding_158/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_158/embedding_lookup/256360992*,
_output_shapes
:??????????*
dtype02 
embedding_158/embedding_lookup?
'embedding_158/embedding_lookup/IdentityIdentity'embedding_158/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_158/embedding_lookup/256360992*,
_output_shapes
:??????????2)
'embedding_158/embedding_lookup/Identity?
)embedding_158/embedding_lookup/Identity_1Identity0embedding_158/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_158/embedding_lookup/Identity_1?
embedding_156/CastCast&tf.compat.v1.floor_div_52/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_156/Cast?
embedding_156/embedding_lookupResourceGather(embedding_156_embedding_lookup_256360998embedding_156/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_156/embedding_lookup/256360998*,
_output_shapes
:??????????*
dtype02 
embedding_156/embedding_lookup?
'embedding_156/embedding_lookup/IdentityIdentity'embedding_156/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_156/embedding_lookup/256360998*,
_output_shapes
:??????????2)
'embedding_156/embedding_lookup/Identity?
)embedding_156/embedding_lookup/Identity_1Identity0embedding_156/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_156/embedding_lookup/Identity_1?
tf.cast_78/CastCast)tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_78/Cast?
tf.__operators__.add_156/AddV2AddV22embedding_158/embedding_lookup/Identity_1:output:02embedding_156/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_156/AddV2?
embedding_157/CastCast tf.math.floormod_52/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_157/Cast?
embedding_157/embedding_lookupResourceGather(embedding_157_embedding_lookup_256361006embedding_157/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_157/embedding_lookup/256361006*,
_output_shapes
:??????????*
dtype02 
embedding_157/embedding_lookup?
'embedding_157/embedding_lookup/IdentityIdentity'embedding_157/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_157/embedding_lookup/256361006*,
_output_shapes
:??????????2)
'embedding_157/embedding_lookup/Identity?
)embedding_157/embedding_lookup/Identity_1Identity0embedding_157/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_157/embedding_lookup/Identity_1?
tf.__operators__.add_157/AddV2AddV2"tf.__operators__.add_156/AddV2:z:02embedding_157/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_157/AddV2?
 tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_52/ExpandDims/dim?
tf.expand_dims_52/ExpandDims
ExpandDimstf.cast_78/Cast:y:0)tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_52/ExpandDims?
tf.math.multiply_52/MulMul"tf.__operators__.add_157/AddV2:z:0%tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_52/Mul?
+tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_52/Sum/reduction_indices?
tf.math.reduce_sum_52/SumSumtf.math.multiply_52/Mul:z:04tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_52/Sum?
IdentityIdentity"tf.math.reduce_sum_52/Sum:output:0^embedding_156/embedding_lookup^embedding_157/embedding_lookup^embedding_158/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2@
embedding_156/embedding_lookupembedding_156/embedding_lookup2@
embedding_157/embedding_lookupembedding_157/embedding_lookup2@
embedding_158/embedding_lookupembedding_158/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360838

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_80_greaterequal_y4
0model_52_tf_math_greater_equal_78_greaterequal_y5
1model_52_embedding_158_embedding_lookup_2563606885
1model_52_embedding_156_embedding_lookup_2563606945
1model_52_embedding_157_embedding_lookup_2563607024
0model_53_tf_math_greater_equal_79_greaterequal_y5
1model_53_embedding_161_embedding_lookup_2563607265
1model_53_embedding_159_embedding_lookup_2563607325
1model_53_embedding_160_embedding_lookup_256360740/
+tf_clip_by_value_80_clip_by_value_minimum_y'
#tf_clip_by_value_80_clip_by_value_y,
(dense_234_matmul_readvariableop_resource-
)dense_234_biasadd_readvariableop_resource,
(dense_237_matmul_readvariableop_resource-
)dense_237_biasadd_readvariableop_resource,
(dense_235_matmul_readvariableop_resource-
)dense_235_biasadd_readvariableop_resource,
(dense_236_matmul_readvariableop_resource-
)dense_236_biasadd_readvariableop_resource,
(dense_238_matmul_readvariableop_resource-
)dense_238_biasadd_readvariableop_resource,
(dense_239_matmul_readvariableop_resource-
)dense_239_biasadd_readvariableop_resource,
(dense_240_matmul_readvariableop_resource-
)dense_240_biasadd_readvariableop_resource,
(dense_241_matmul_readvariableop_resource-
)dense_241_biasadd_readvariableop_resourceA
=normalize_26_normalization_26_reshape_readvariableop_resourceC
?normalize_26_normalization_26_reshape_1_readvariableop_resource,
(dense_242_matmul_readvariableop_resource-
)dense_242_biasadd_readvariableop_resource
identity?? dense_234/BiasAdd/ReadVariableOp?dense_234/MatMul/ReadVariableOp? dense_235/BiasAdd/ReadVariableOp?dense_235/MatMul/ReadVariableOp? dense_236/BiasAdd/ReadVariableOp?dense_236/MatMul/ReadVariableOp? dense_237/BiasAdd/ReadVariableOp?dense_237/MatMul/ReadVariableOp? dense_238/BiasAdd/ReadVariableOp?dense_238/MatMul/ReadVariableOp? dense_239/BiasAdd/ReadVariableOp?dense_239/MatMul/ReadVariableOp? dense_240/BiasAdd/ReadVariableOp?dense_240/MatMul/ReadVariableOp? dense_241/BiasAdd/ReadVariableOp?dense_241/MatMul/ReadVariableOp? dense_242/BiasAdd/ReadVariableOp?dense_242/MatMul/ReadVariableOp?'model_52/embedding_156/embedding_lookup?'model_52/embedding_157/embedding_lookup?'model_52/embedding_158/embedding_lookup?'model_53/embedding_159/embedding_lookup?'model_53/embedding_160/embedding_lookup?'model_53/embedding_161/embedding_lookup?4normalize_26/normalization_26/Reshape/ReadVariableOp?6normalize_26/normalization_26/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_80/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_80/GreaterEqual?
model_52/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_52/flatten_52/Const?
model_52/flatten_52/ReshapeReshape
inputs_0_0"model_52/flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2
model_52/flatten_52/Reshape?
4model_52/tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_52/tf.clip_by_value_78/clip_by_value/Minimum/y?
2model_52/tf.clip_by_value_78/clip_by_value/MinimumMinimum$model_52/flatten_52/Reshape:output:0=model_52/tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_52/tf.clip_by_value_78/clip_by_value/Minimum?
,model_52/tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_52/tf.clip_by_value_78/clip_by_value/y?
*model_52/tf.clip_by_value_78/clip_by_valueMaximum6model_52/tf.clip_by_value_78/clip_by_value/Minimum:z:05model_52/tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_52/tf.clip_by_value_78/clip_by_value?
-model_52/tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_52/tf.compat.v1.floor_div_52/FloorDiv/y?
+model_52/tf.compat.v1.floor_div_52/FloorDivFloorDiv.model_52/tf.clip_by_value_78/clip_by_value:z:06model_52/tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_52/tf.compat.v1.floor_div_52/FloorDiv?
.model_52/tf.math.greater_equal_78/GreaterEqualGreaterEqual$model_52/flatten_52/Reshape:output:00model_52_tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_52/tf.math.greater_equal_78/GreaterEqual?
'model_52/tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_52/tf.math.floormod_52/FloorMod/y?
%model_52/tf.math.floormod_52/FloorModFloorMod.model_52/tf.clip_by_value_78/clip_by_value:z:00model_52/tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_52/tf.math.floormod_52/FloorMod?
model_52/embedding_158/CastCast.model_52/tf.clip_by_value_78/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_52/embedding_158/Cast?
'model_52/embedding_158/embedding_lookupResourceGather1model_52_embedding_158_embedding_lookup_256360688model_52/embedding_158/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_52/embedding_158/embedding_lookup/256360688*,
_output_shapes
:??????????*
dtype02)
'model_52/embedding_158/embedding_lookup?
0model_52/embedding_158/embedding_lookup/IdentityIdentity0model_52/embedding_158/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_52/embedding_158/embedding_lookup/256360688*,
_output_shapes
:??????????22
0model_52/embedding_158/embedding_lookup/Identity?
2model_52/embedding_158/embedding_lookup/Identity_1Identity9model_52/embedding_158/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_52/embedding_158/embedding_lookup/Identity_1?
model_52/embedding_156/CastCast/model_52/tf.compat.v1.floor_div_52/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_52/embedding_156/Cast?
'model_52/embedding_156/embedding_lookupResourceGather1model_52_embedding_156_embedding_lookup_256360694model_52/embedding_156/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_52/embedding_156/embedding_lookup/256360694*,
_output_shapes
:??????????*
dtype02)
'model_52/embedding_156/embedding_lookup?
0model_52/embedding_156/embedding_lookup/IdentityIdentity0model_52/embedding_156/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_52/embedding_156/embedding_lookup/256360694*,
_output_shapes
:??????????22
0model_52/embedding_156/embedding_lookup/Identity?
2model_52/embedding_156/embedding_lookup/Identity_1Identity9model_52/embedding_156/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_52/embedding_156/embedding_lookup/Identity_1?
model_52/tf.cast_78/CastCast2model_52/tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_52/tf.cast_78/Cast?
'model_52/tf.__operators__.add_156/AddV2AddV2;model_52/embedding_158/embedding_lookup/Identity_1:output:0;model_52/embedding_156/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_52/tf.__operators__.add_156/AddV2?
model_52/embedding_157/CastCast)model_52/tf.math.floormod_52/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_52/embedding_157/Cast?
'model_52/embedding_157/embedding_lookupResourceGather1model_52_embedding_157_embedding_lookup_256360702model_52/embedding_157/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_52/embedding_157/embedding_lookup/256360702*,
_output_shapes
:??????????*
dtype02)
'model_52/embedding_157/embedding_lookup?
0model_52/embedding_157/embedding_lookup/IdentityIdentity0model_52/embedding_157/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_52/embedding_157/embedding_lookup/256360702*,
_output_shapes
:??????????22
0model_52/embedding_157/embedding_lookup/Identity?
2model_52/embedding_157/embedding_lookup/Identity_1Identity9model_52/embedding_157/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_52/embedding_157/embedding_lookup/Identity_1?
'model_52/tf.__operators__.add_157/AddV2AddV2+model_52/tf.__operators__.add_156/AddV2:z:0;model_52/embedding_157/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_52/tf.__operators__.add_157/AddV2?
)model_52/tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_52/tf.expand_dims_52/ExpandDims/dim?
%model_52/tf.expand_dims_52/ExpandDims
ExpandDimsmodel_52/tf.cast_78/Cast:y:02model_52/tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_52/tf.expand_dims_52/ExpandDims?
 model_52/tf.math.multiply_52/MulMul+model_52/tf.__operators__.add_157/AddV2:z:0.model_52/tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_52/tf.math.multiply_52/Mul?
4model_52/tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_52/tf.math.reduce_sum_52/Sum/reduction_indices?
"model_52/tf.math.reduce_sum_52/SumSum$model_52/tf.math.multiply_52/Mul:z:0=model_52/tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_52/tf.math.reduce_sum_52/Sum?
model_53/flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_53/flatten_53/Const?
model_53/flatten_53/ReshapeReshape
inputs_0_1"model_53/flatten_53/Const:output:0*
T0*'
_output_shapes
:?????????2
model_53/flatten_53/Reshape?
4model_53/tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_53/tf.clip_by_value_79/clip_by_value/Minimum/y?
2model_53/tf.clip_by_value_79/clip_by_value/MinimumMinimum$model_53/flatten_53/Reshape:output:0=model_53/tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_53/tf.clip_by_value_79/clip_by_value/Minimum?
,model_53/tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_53/tf.clip_by_value_79/clip_by_value/y?
*model_53/tf.clip_by_value_79/clip_by_valueMaximum6model_53/tf.clip_by_value_79/clip_by_value/Minimum:z:05model_53/tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_53/tf.clip_by_value_79/clip_by_value?
-model_53/tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_53/tf.compat.v1.floor_div_53/FloorDiv/y?
+model_53/tf.compat.v1.floor_div_53/FloorDivFloorDiv.model_53/tf.clip_by_value_79/clip_by_value:z:06model_53/tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_53/tf.compat.v1.floor_div_53/FloorDiv?
.model_53/tf.math.greater_equal_79/GreaterEqualGreaterEqual$model_53/flatten_53/Reshape:output:00model_53_tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_53/tf.math.greater_equal_79/GreaterEqual?
'model_53/tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_53/tf.math.floormod_53/FloorMod/y?
%model_53/tf.math.floormod_53/FloorModFloorMod.model_53/tf.clip_by_value_79/clip_by_value:z:00model_53/tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_53/tf.math.floormod_53/FloorMod?
model_53/embedding_161/CastCast.model_53/tf.clip_by_value_79/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_53/embedding_161/Cast?
'model_53/embedding_161/embedding_lookupResourceGather1model_53_embedding_161_embedding_lookup_256360726model_53/embedding_161/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_53/embedding_161/embedding_lookup/256360726*,
_output_shapes
:??????????*
dtype02)
'model_53/embedding_161/embedding_lookup?
0model_53/embedding_161/embedding_lookup/IdentityIdentity0model_53/embedding_161/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_53/embedding_161/embedding_lookup/256360726*,
_output_shapes
:??????????22
0model_53/embedding_161/embedding_lookup/Identity?
2model_53/embedding_161/embedding_lookup/Identity_1Identity9model_53/embedding_161/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_53/embedding_161/embedding_lookup/Identity_1?
model_53/embedding_159/CastCast/model_53/tf.compat.v1.floor_div_53/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_53/embedding_159/Cast?
'model_53/embedding_159/embedding_lookupResourceGather1model_53_embedding_159_embedding_lookup_256360732model_53/embedding_159/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_53/embedding_159/embedding_lookup/256360732*,
_output_shapes
:??????????*
dtype02)
'model_53/embedding_159/embedding_lookup?
0model_53/embedding_159/embedding_lookup/IdentityIdentity0model_53/embedding_159/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_53/embedding_159/embedding_lookup/256360732*,
_output_shapes
:??????????22
0model_53/embedding_159/embedding_lookup/Identity?
2model_53/embedding_159/embedding_lookup/Identity_1Identity9model_53/embedding_159/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_53/embedding_159/embedding_lookup/Identity_1?
model_53/tf.cast_79/CastCast2model_53/tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_53/tf.cast_79/Cast?
'model_53/tf.__operators__.add_158/AddV2AddV2;model_53/embedding_161/embedding_lookup/Identity_1:output:0;model_53/embedding_159/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_53/tf.__operators__.add_158/AddV2?
model_53/embedding_160/CastCast)model_53/tf.math.floormod_53/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_53/embedding_160/Cast?
'model_53/embedding_160/embedding_lookupResourceGather1model_53_embedding_160_embedding_lookup_256360740model_53/embedding_160/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_53/embedding_160/embedding_lookup/256360740*,
_output_shapes
:??????????*
dtype02)
'model_53/embedding_160/embedding_lookup?
0model_53/embedding_160/embedding_lookup/IdentityIdentity0model_53/embedding_160/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_53/embedding_160/embedding_lookup/256360740*,
_output_shapes
:??????????22
0model_53/embedding_160/embedding_lookup/Identity?
2model_53/embedding_160/embedding_lookup/Identity_1Identity9model_53/embedding_160/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????24
2model_53/embedding_160/embedding_lookup/Identity_1?
'model_53/tf.__operators__.add_159/AddV2AddV2+model_53/tf.__operators__.add_158/AddV2:z:0;model_53/embedding_160/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2)
'model_53/tf.__operators__.add_159/AddV2?
)model_53/tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_53/tf.expand_dims_53/ExpandDims/dim?
%model_53/tf.expand_dims_53/ExpandDims
ExpandDimsmodel_53/tf.cast_79/Cast:y:02model_53/tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_53/tf.expand_dims_53/ExpandDims?
 model_53/tf.math.multiply_53/MulMul+model_53/tf.__operators__.add_159/AddV2:z:0.model_53/tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_53/tf.math.multiply_53/Mul?
4model_53/tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_53/tf.math.reduce_sum_53/Sum/reduction_indices?
"model_53/tf.math.reduce_sum_53/SumSum$model_53/tf.math.multiply_53/Mul:z:0=model_53/tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_53/tf.math.reduce_sum_53/Sum?
)tf.clip_by_value_80/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_80/clip_by_value/Minimum?
!tf.clip_by_value_80/clip_by_valueMaximum-tf.clip_by_value_80/clip_by_value/Minimum:z:0#tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_80/clip_by_value?
tf.cast_80/CastCast)tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_80/Castv
tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_78/concat/axis?
tf.concat_78/concatConcatV2+model_52/tf.math.reduce_sum_52/Sum:output:0+model_53/tf.math.reduce_sum_53/Sum:output:0!tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_78/concat
tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_79/concat/axis?
tf.concat_79/concatConcatV2%tf.clip_by_value_80/clip_by_value:z:0tf.cast_80/Cast:y:0!tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_79/concat?
dense_234/MatMul/ReadVariableOpReadVariableOp(dense_234_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_234/MatMul/ReadVariableOp?
dense_234/MatMulMatMultf.concat_78/concat:output:0'dense_234/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_234/MatMul?
 dense_234/BiasAdd/ReadVariableOpReadVariableOp)dense_234_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_234/BiasAdd/ReadVariableOp?
dense_234/BiasAddBiasAdddense_234/MatMul:product:0(dense_234/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_234/BiasAddw
dense_234/ReluReludense_234/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_234/Relu?
dense_237/MatMul/ReadVariableOpReadVariableOp(dense_237_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_237/MatMul/ReadVariableOp?
dense_237/MatMulMatMultf.concat_79/concat:output:0'dense_237/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_237/MatMul?
 dense_237/BiasAdd/ReadVariableOpReadVariableOp)dense_237_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_237/BiasAdd/ReadVariableOp?
dense_237/BiasAddBiasAdddense_237/MatMul:product:0(dense_237/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_237/BiasAdd?
dense_235/MatMul/ReadVariableOpReadVariableOp(dense_235_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_235/MatMul/ReadVariableOp?
dense_235/MatMulMatMuldense_234/Relu:activations:0'dense_235/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_235/MatMul?
 dense_235/BiasAdd/ReadVariableOpReadVariableOp)dense_235_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_235/BiasAdd/ReadVariableOp?
dense_235/BiasAddBiasAdddense_235/MatMul:product:0(dense_235/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_235/BiasAddw
dense_235/ReluReludense_235/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_235/Relu?
dense_236/MatMul/ReadVariableOpReadVariableOp(dense_236_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_236/MatMul/ReadVariableOp?
dense_236/MatMulMatMuldense_235/Relu:activations:0'dense_236/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_236/MatMul?
 dense_236/BiasAdd/ReadVariableOpReadVariableOp)dense_236_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_236/BiasAdd/ReadVariableOp?
dense_236/BiasAddBiasAdddense_236/MatMul:product:0(dense_236/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_236/BiasAddw
dense_236/ReluReludense_236/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_236/Relu?
dense_238/MatMul/ReadVariableOpReadVariableOp(dense_238_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_238/MatMul/ReadVariableOp?
dense_238/MatMulMatMuldense_237/BiasAdd:output:0'dense_238/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_238/MatMul?
 dense_238/BiasAdd/ReadVariableOpReadVariableOp)dense_238_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_238/BiasAdd/ReadVariableOp?
dense_238/BiasAddBiasAdddense_238/MatMul:product:0(dense_238/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_238/BiasAdd
tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_80/concat/axis?
tf.concat_80/concatConcatV2dense_236/Relu:activations:0dense_238/BiasAdd:output:0!tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_80/concat?
dense_239/MatMul/ReadVariableOpReadVariableOp(dense_239_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_239/MatMul/ReadVariableOp?
dense_239/MatMulMatMultf.concat_80/concat:output:0'dense_239/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_239/MatMul?
 dense_239/BiasAdd/ReadVariableOpReadVariableOp)dense_239_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_239/BiasAdd/ReadVariableOp?
dense_239/BiasAddBiasAdddense_239/MatMul:product:0(dense_239/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_239/BiasAdd
tf.nn.relu_78/ReluReludense_239/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_78/Relu?
dense_240/MatMul/ReadVariableOpReadVariableOp(dense_240_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_240/MatMul/ReadVariableOp?
dense_240/MatMulMatMul tf.nn.relu_78/Relu:activations:0'dense_240/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_240/MatMul?
 dense_240/BiasAdd/ReadVariableOpReadVariableOp)dense_240_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_240/BiasAdd/ReadVariableOp?
dense_240/BiasAddBiasAdddense_240/MatMul:product:0(dense_240/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_240/BiasAdd?
tf.__operators__.add_160/AddV2AddV2dense_240/BiasAdd:output:0 tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_160/AddV2?
tf.nn.relu_79/ReluRelu"tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_79/Relu?
dense_241/MatMul/ReadVariableOpReadVariableOp(dense_241_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_241/MatMul/ReadVariableOp?
dense_241/MatMulMatMul tf.nn.relu_79/Relu:activations:0'dense_241/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_241/MatMul?
 dense_241/BiasAdd/ReadVariableOpReadVariableOp)dense_241_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_241/BiasAdd/ReadVariableOp?
dense_241/BiasAddBiasAdddense_241/MatMul:product:0(dense_241/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_241/BiasAdd?
tf.__operators__.add_161/AddV2AddV2dense_241/BiasAdd:output:0 tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_161/AddV2?
tf.nn.relu_80/ReluRelu"tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_80/Relu?
4normalize_26/normalization_26/Reshape/ReadVariableOpReadVariableOp=normalize_26_normalization_26_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_26/normalization_26/Reshape/ReadVariableOp?
+normalize_26/normalization_26/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_26/normalization_26/Reshape/shape?
%normalize_26/normalization_26/ReshapeReshape<normalize_26/normalization_26/Reshape/ReadVariableOp:value:04normalize_26/normalization_26/Reshape/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_26/normalization_26/Reshape?
6normalize_26/normalization_26/Reshape_1/ReadVariableOpReadVariableOp?normalize_26_normalization_26_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6normalize_26/normalization_26/Reshape_1/ReadVariableOp?
-normalize_26/normalization_26/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2/
-normalize_26/normalization_26/Reshape_1/shape?
'normalize_26/normalization_26/Reshape_1Reshape>normalize_26/normalization_26/Reshape_1/ReadVariableOp:value:06normalize_26/normalization_26/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2)
'normalize_26/normalization_26/Reshape_1?
!normalize_26/normalization_26/subSub tf.nn.relu_80/Relu:activations:0.normalize_26/normalization_26/Reshape:output:0*
T0*(
_output_shapes
:??????????2#
!normalize_26/normalization_26/sub?
"normalize_26/normalization_26/SqrtSqrt0normalize_26/normalization_26/Reshape_1:output:0*
T0*
_output_shapes
:	?2$
"normalize_26/normalization_26/Sqrt?
'normalize_26/normalization_26/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32)
'normalize_26/normalization_26/Maximum/y?
%normalize_26/normalization_26/MaximumMaximum&normalize_26/normalization_26/Sqrt:y:00normalize_26/normalization_26/Maximum/y:output:0*
T0*
_output_shapes
:	?2'
%normalize_26/normalization_26/Maximum?
%normalize_26/normalization_26/truedivRealDiv%normalize_26/normalization_26/sub:z:0)normalize_26/normalization_26/Maximum:z:0*
T0*(
_output_shapes
:??????????2'
%normalize_26/normalization_26/truediv?
dense_242/MatMul/ReadVariableOpReadVariableOp(dense_242_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_242/MatMul/ReadVariableOp?
dense_242/MatMulMatMul)normalize_26/normalization_26/truediv:z:0'dense_242/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_242/MatMul?
 dense_242/BiasAdd/ReadVariableOpReadVariableOp)dense_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_242/BiasAdd/ReadVariableOp?
dense_242/BiasAddBiasAdddense_242/MatMul:product:0(dense_242/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_242/BiasAdd?
IdentityIdentitydense_242/BiasAdd:output:0!^dense_234/BiasAdd/ReadVariableOp ^dense_234/MatMul/ReadVariableOp!^dense_235/BiasAdd/ReadVariableOp ^dense_235/MatMul/ReadVariableOp!^dense_236/BiasAdd/ReadVariableOp ^dense_236/MatMul/ReadVariableOp!^dense_237/BiasAdd/ReadVariableOp ^dense_237/MatMul/ReadVariableOp!^dense_238/BiasAdd/ReadVariableOp ^dense_238/MatMul/ReadVariableOp!^dense_239/BiasAdd/ReadVariableOp ^dense_239/MatMul/ReadVariableOp!^dense_240/BiasAdd/ReadVariableOp ^dense_240/MatMul/ReadVariableOp!^dense_241/BiasAdd/ReadVariableOp ^dense_241/MatMul/ReadVariableOp!^dense_242/BiasAdd/ReadVariableOp ^dense_242/MatMul/ReadVariableOp(^model_52/embedding_156/embedding_lookup(^model_52/embedding_157/embedding_lookup(^model_52/embedding_158/embedding_lookup(^model_53/embedding_159/embedding_lookup(^model_53/embedding_160/embedding_lookup(^model_53/embedding_161/embedding_lookup5^normalize_26/normalization_26/Reshape/ReadVariableOp7^normalize_26/normalization_26/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_234/BiasAdd/ReadVariableOp dense_234/BiasAdd/ReadVariableOp2B
dense_234/MatMul/ReadVariableOpdense_234/MatMul/ReadVariableOp2D
 dense_235/BiasAdd/ReadVariableOp dense_235/BiasAdd/ReadVariableOp2B
dense_235/MatMul/ReadVariableOpdense_235/MatMul/ReadVariableOp2D
 dense_236/BiasAdd/ReadVariableOp dense_236/BiasAdd/ReadVariableOp2B
dense_236/MatMul/ReadVariableOpdense_236/MatMul/ReadVariableOp2D
 dense_237/BiasAdd/ReadVariableOp dense_237/BiasAdd/ReadVariableOp2B
dense_237/MatMul/ReadVariableOpdense_237/MatMul/ReadVariableOp2D
 dense_238/BiasAdd/ReadVariableOp dense_238/BiasAdd/ReadVariableOp2B
dense_238/MatMul/ReadVariableOpdense_238/MatMul/ReadVariableOp2D
 dense_239/BiasAdd/ReadVariableOp dense_239/BiasAdd/ReadVariableOp2B
dense_239/MatMul/ReadVariableOpdense_239/MatMul/ReadVariableOp2D
 dense_240/BiasAdd/ReadVariableOp dense_240/BiasAdd/ReadVariableOp2B
dense_240/MatMul/ReadVariableOpdense_240/MatMul/ReadVariableOp2D
 dense_241/BiasAdd/ReadVariableOp dense_241/BiasAdd/ReadVariableOp2B
dense_241/MatMul/ReadVariableOpdense_241/MatMul/ReadVariableOp2D
 dense_242/BiasAdd/ReadVariableOp dense_242/BiasAdd/ReadVariableOp2B
dense_242/MatMul/ReadVariableOpdense_242/MatMul/ReadVariableOp2R
'model_52/embedding_156/embedding_lookup'model_52/embedding_156/embedding_lookup2R
'model_52/embedding_157/embedding_lookup'model_52/embedding_157/embedding_lookup2R
'model_52/embedding_158/embedding_lookup'model_52/embedding_158/embedding_lookup2R
'model_53/embedding_159/embedding_lookup'model_53/embedding_159/embedding_lookup2R
'model_53/embedding_160/embedding_lookup'model_53/embedding_160/embedding_lookup2R
'model_53/embedding_161/embedding_lookup'model_53/embedding_161/embedding_lookup2l
4normalize_26/normalization_26/Reshape/ReadVariableOp4normalize_26/normalization_26/Reshape/ReadVariableOp2p
6normalize_26/normalization_26/Reshape_1/ReadVariableOp6normalize_26/normalization_26/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?:
?
G__inference_model_52_layer_call_and_return_conditional_losses_256361060

inputs+
'tf_math_greater_equal_78_greaterequal_y,
(embedding_158_embedding_lookup_256361034,
(embedding_156_embedding_lookup_256361040,
(embedding_157_embedding_lookup_256361048
identity??embedding_156/embedding_lookup?embedding_157/embedding_lookup?embedding_158/embedding_lookupu
flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_52/Const?
flatten_52/ReshapeReshapeinputsflatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_52/Reshape?
+tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_78/clip_by_value/Minimum/y?
)tf.clip_by_value_78/clip_by_value/MinimumMinimumflatten_52/Reshape:output:04tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_78/clip_by_value/Minimum?
#tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_78/clip_by_value/y?
!tf.clip_by_value_78/clip_by_valueMaximum-tf.clip_by_value_78/clip_by_value/Minimum:z:0,tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_78/clip_by_value?
$tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_52/FloorDiv/y?
"tf.compat.v1.floor_div_52/FloorDivFloorDiv%tf.clip_by_value_78/clip_by_value:z:0-tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_52/FloorDiv?
%tf.math.greater_equal_78/GreaterEqualGreaterEqualflatten_52/Reshape:output:0'tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_78/GreaterEqual?
tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_52/FloorMod/y?
tf.math.floormod_52/FloorModFloorMod%tf.clip_by_value_78/clip_by_value:z:0'tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_52/FloorMod?
embedding_158/CastCast%tf.clip_by_value_78/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_158/Cast?
embedding_158/embedding_lookupResourceGather(embedding_158_embedding_lookup_256361034embedding_158/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_158/embedding_lookup/256361034*,
_output_shapes
:??????????*
dtype02 
embedding_158/embedding_lookup?
'embedding_158/embedding_lookup/IdentityIdentity'embedding_158/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_158/embedding_lookup/256361034*,
_output_shapes
:??????????2)
'embedding_158/embedding_lookup/Identity?
)embedding_158/embedding_lookup/Identity_1Identity0embedding_158/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_158/embedding_lookup/Identity_1?
embedding_156/CastCast&tf.compat.v1.floor_div_52/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_156/Cast?
embedding_156/embedding_lookupResourceGather(embedding_156_embedding_lookup_256361040embedding_156/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_156/embedding_lookup/256361040*,
_output_shapes
:??????????*
dtype02 
embedding_156/embedding_lookup?
'embedding_156/embedding_lookup/IdentityIdentity'embedding_156/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_156/embedding_lookup/256361040*,
_output_shapes
:??????????2)
'embedding_156/embedding_lookup/Identity?
)embedding_156/embedding_lookup/Identity_1Identity0embedding_156/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_156/embedding_lookup/Identity_1?
tf.cast_78/CastCast)tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_78/Cast?
tf.__operators__.add_156/AddV2AddV22embedding_158/embedding_lookup/Identity_1:output:02embedding_156/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_156/AddV2?
embedding_157/CastCast tf.math.floormod_52/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_157/Cast?
embedding_157/embedding_lookupResourceGather(embedding_157_embedding_lookup_256361048embedding_157/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_157/embedding_lookup/256361048*,
_output_shapes
:??????????*
dtype02 
embedding_157/embedding_lookup?
'embedding_157/embedding_lookup/IdentityIdentity'embedding_157/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_157/embedding_lookup/256361048*,
_output_shapes
:??????????2)
'embedding_157/embedding_lookup/Identity?
)embedding_157/embedding_lookup/Identity_1Identity0embedding_157/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_157/embedding_lookup/Identity_1?
tf.__operators__.add_157/AddV2AddV2"tf.__operators__.add_156/AddV2:z:02embedding_157/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_157/AddV2?
 tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_52/ExpandDims/dim?
tf.expand_dims_52/ExpandDims
ExpandDimstf.cast_78/Cast:y:0)tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_52/ExpandDims?
tf.math.multiply_52/MulMul"tf.__operators__.add_157/AddV2:z:0%tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_52/Mul?
+tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_52/Sum/reduction_indices?
tf.math.reduce_sum_52/SumSumtf.math.multiply_52/Mul:z:04tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_52/Sum?
IdentityIdentity"tf.math.reduce_sum_52/Sum:output:0^embedding_156/embedding_lookup^embedding_157/embedding_lookup^embedding_158/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2@
embedding_156/embedding_lookupembedding_156/embedding_lookup2@
embedding_157/embedding_lookupembedding_157/embedding_lookup2@
embedding_158/embedding_lookupembedding_158/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_model_53_layer_call_fn_256359639
input_54
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_54unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563596282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_54:

_output_shapes
: 
?	
?
H__inference_dense_237_layer_call_and_return_conditional_losses_256361246

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_161_layer_call_and_return_conditional_losses_256359451

inputs
embedding_lookup_256359445
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256359445Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256359445*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256359445*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_238_layer_call_and_return_conditional_losses_256361285

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_159_layer_call_and_return_conditional_losses_256361496

inputs
embedding_lookup_256361490
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256361490Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256361490*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256361490*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
G__inference_model_53_layer_call_and_return_conditional_losses_256359548
input_54+
'tf_math_greater_equal_79_greaterequal_y
embedding_161_256359530
embedding_159_256359533
embedding_160_256359538
identity??%embedding_159/StatefulPartitionedCall?%embedding_160/StatefulPartitionedCall?%embedding_161/StatefulPartitionedCall?
flatten_53/PartitionedCallPartitionedCallinput_54*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_53_layer_call_and_return_conditional_losses_2563594232
flatten_53/PartitionedCall?
+tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_79/clip_by_value/Minimum/y?
)tf.clip_by_value_79/clip_by_value/MinimumMinimum#flatten_53/PartitionedCall:output:04tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_79/clip_by_value/Minimum?
#tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_79/clip_by_value/y?
!tf.clip_by_value_79/clip_by_valueMaximum-tf.clip_by_value_79/clip_by_value/Minimum:z:0,tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_79/clip_by_value?
$tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_53/FloorDiv/y?
"tf.compat.v1.floor_div_53/FloorDivFloorDiv%tf.clip_by_value_79/clip_by_value:z:0-tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_53/FloorDiv?
%tf.math.greater_equal_79/GreaterEqualGreaterEqual#flatten_53/PartitionedCall:output:0'tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_79/GreaterEqual?
tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_53/FloorMod/y?
tf.math.floormod_53/FloorModFloorMod%tf.clip_by_value_79/clip_by_value:z:0'tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_53/FloorMod?
%embedding_161/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_79/clip_by_value:z:0embedding_161_256359530*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_161_layer_call_and_return_conditional_losses_2563594512'
%embedding_161/StatefulPartitionedCall?
%embedding_159/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_53/FloorDiv:z:0embedding_159_256359533*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_159_layer_call_and_return_conditional_losses_2563594732'
%embedding_159/StatefulPartitionedCall?
tf.cast_79/CastCast)tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_79/Cast?
tf.__operators__.add_158/AddV2AddV2.embedding_161/StatefulPartitionedCall:output:0.embedding_159/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_158/AddV2?
%embedding_160/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_53/FloorMod:z:0embedding_160_256359538*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_160_layer_call_and_return_conditional_losses_2563594972'
%embedding_160/StatefulPartitionedCall?
tf.__operators__.add_159/AddV2AddV2"tf.__operators__.add_158/AddV2:z:0.embedding_160/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_159/AddV2?
 tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_53/ExpandDims/dim?
tf.expand_dims_53/ExpandDims
ExpandDimstf.cast_79/Cast:y:0)tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_53/ExpandDims?
tf.math.multiply_53/MulMul"tf.__operators__.add_159/AddV2:z:0%tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_53/Mul?
+tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_53/Sum/reduction_indices?
tf.math.reduce_sum_53/SumSumtf.math.multiply_53/Mul:z:04tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_53/Sum?
IdentityIdentity"tf.math.reduce_sum_53/Sum:output:0&^embedding_159/StatefulPartitionedCall&^embedding_160/StatefulPartitionedCall&^embedding_161/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_159/StatefulPartitionedCall%embedding_159/StatefulPartitionedCall2N
%embedding_160/StatefulPartitionedCall%embedding_160/StatefulPartitionedCall2N
%embedding_161/StatefulPartitionedCall%embedding_161/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_54:

_output_shapes
: 
?.
?
G__inference_model_53_layer_call_and_return_conditional_losses_256359516
input_54+
'tf_math_greater_equal_79_greaterequal_y
embedding_161_256359460
embedding_159_256359482
embedding_160_256359506
identity??%embedding_159/StatefulPartitionedCall?%embedding_160/StatefulPartitionedCall?%embedding_161/StatefulPartitionedCall?
flatten_53/PartitionedCallPartitionedCallinput_54*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_53_layer_call_and_return_conditional_losses_2563594232
flatten_53/PartitionedCall?
+tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_79/clip_by_value/Minimum/y?
)tf.clip_by_value_79/clip_by_value/MinimumMinimum#flatten_53/PartitionedCall:output:04tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_79/clip_by_value/Minimum?
#tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_79/clip_by_value/y?
!tf.clip_by_value_79/clip_by_valueMaximum-tf.clip_by_value_79/clip_by_value/Minimum:z:0,tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_79/clip_by_value?
$tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_53/FloorDiv/y?
"tf.compat.v1.floor_div_53/FloorDivFloorDiv%tf.clip_by_value_79/clip_by_value:z:0-tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_53/FloorDiv?
%tf.math.greater_equal_79/GreaterEqualGreaterEqual#flatten_53/PartitionedCall:output:0'tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_79/GreaterEqual?
tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_53/FloorMod/y?
tf.math.floormod_53/FloorModFloorMod%tf.clip_by_value_79/clip_by_value:z:0'tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_53/FloorMod?
%embedding_161/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_79/clip_by_value:z:0embedding_161_256359460*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_161_layer_call_and_return_conditional_losses_2563594512'
%embedding_161/StatefulPartitionedCall?
%embedding_159/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_53/FloorDiv:z:0embedding_159_256359482*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_159_layer_call_and_return_conditional_losses_2563594732'
%embedding_159/StatefulPartitionedCall?
tf.cast_79/CastCast)tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_79/Cast?
tf.__operators__.add_158/AddV2AddV2.embedding_161/StatefulPartitionedCall:output:0.embedding_159/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_158/AddV2?
%embedding_160/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_53/FloorMod:z:0embedding_160_256359506*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_160_layer_call_and_return_conditional_losses_2563594972'
%embedding_160/StatefulPartitionedCall?
tf.__operators__.add_159/AddV2AddV2"tf.__operators__.add_158/AddV2:z:0.embedding_160/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_159/AddV2?
 tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_53/ExpandDims/dim?
tf.expand_dims_53/ExpandDims
ExpandDimstf.cast_79/Cast:y:0)tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_53/ExpandDims?
tf.math.multiply_53/MulMul"tf.__operators__.add_159/AddV2:z:0%tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_53/Mul?
+tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_53/Sum/reduction_indices?
tf.math.reduce_sum_53/SumSumtf.math.multiply_53/Mul:z:04tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_53/Sum?
IdentityIdentity"tf.math.reduce_sum_53/Sum:output:0&^embedding_159/StatefulPartitionedCall&^embedding_160/StatefulPartitionedCall&^embedding_161/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_159/StatefulPartitionedCall%embedding_159/StatefulPartitionedCall2N
%embedding_160/StatefulPartitionedCall%embedding_160/StatefulPartitionedCall2N
%embedding_161/StatefulPartitionedCall%embedding_161/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_54:

_output_shapes
: 
?
J
.__inference_flatten_52_layer_call_fn_256361407

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_52_layer_call_and_return_conditional_losses_2563591972
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_signature_wrapper_256360498
bets

cards0

cards1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference__wrapped_model_2563591872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????
:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????


_user_specified_namebets:OK
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_dense_239_layer_call_fn_256361313

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_239_layer_call_and_return_conditional_losses_2563598712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_158_layer_call_and_return_conditional_losses_256359225

inputs
embedding_lookup_256359219
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256359219Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256359219*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256359219*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_242_layer_call_and_return_conditional_losses_256359987

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_239_layer_call_and_return_conditional_losses_256361304

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_normalize_26_layer_call_fn_256361377
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_normalize_26_layer_call_and_return_conditional_losses_2563599612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
H__inference_dense_235_layer_call_and_return_conditional_losses_256361227

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_flatten_52_layer_call_and_return_conditional_losses_256361402

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dense_236_layer_call_fn_256361275

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_236_layer_call_and_return_conditional_losses_2563598172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_239_layer_call_and_return_conditional_losses_256359871

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_240_layer_call_and_return_conditional_losses_256361323

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_158_layer_call_and_return_conditional_losses_256361417

inputs
embedding_lookup_256361411
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256361411Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256361411*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256361411*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
G__inference_model_53_layer_call_and_return_conditional_losses_256359628

inputs+
'tf_math_greater_equal_79_greaterequal_y
embedding_161_256359610
embedding_159_256359613
embedding_160_256359618
identity??%embedding_159/StatefulPartitionedCall?%embedding_160/StatefulPartitionedCall?%embedding_161/StatefulPartitionedCall?
flatten_53/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_53_layer_call_and_return_conditional_losses_2563594232
flatten_53/PartitionedCall?
+tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_79/clip_by_value/Minimum/y?
)tf.clip_by_value_79/clip_by_value/MinimumMinimum#flatten_53/PartitionedCall:output:04tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_79/clip_by_value/Minimum?
#tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_79/clip_by_value/y?
!tf.clip_by_value_79/clip_by_valueMaximum-tf.clip_by_value_79/clip_by_value/Minimum:z:0,tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_79/clip_by_value?
$tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_53/FloorDiv/y?
"tf.compat.v1.floor_div_53/FloorDivFloorDiv%tf.clip_by_value_79/clip_by_value:z:0-tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_53/FloorDiv?
%tf.math.greater_equal_79/GreaterEqualGreaterEqual#flatten_53/PartitionedCall:output:0'tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_79/GreaterEqual?
tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_53/FloorMod/y?
tf.math.floormod_53/FloorModFloorMod%tf.clip_by_value_79/clip_by_value:z:0'tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_53/FloorMod?
%embedding_161/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_79/clip_by_value:z:0embedding_161_256359610*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_161_layer_call_and_return_conditional_losses_2563594512'
%embedding_161/StatefulPartitionedCall?
%embedding_159/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_53/FloorDiv:z:0embedding_159_256359613*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_159_layer_call_and_return_conditional_losses_2563594732'
%embedding_159/StatefulPartitionedCall?
tf.cast_79/CastCast)tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_79/Cast?
tf.__operators__.add_158/AddV2AddV2.embedding_161/StatefulPartitionedCall:output:0.embedding_159/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_158/AddV2?
%embedding_160/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_53/FloorMod:z:0embedding_160_256359618*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_160_layer_call_and_return_conditional_losses_2563594972'
%embedding_160/StatefulPartitionedCall?
tf.__operators__.add_159/AddV2AddV2"tf.__operators__.add_158/AddV2:z:0.embedding_160/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_159/AddV2?
 tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_53/ExpandDims/dim?
tf.expand_dims_53/ExpandDims
ExpandDimstf.cast_79/Cast:y:0)tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_53/ExpandDims?
tf.math.multiply_53/MulMul"tf.__operators__.add_159/AddV2:z:0%tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_53/Mul?
+tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_53/Sum/reduction_indices?
tf.math.reduce_sum_53/SumSumtf.math.multiply_53/Mul:z:04tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_53/Sum?
IdentityIdentity"tf.math.reduce_sum_53/Sum:output:0&^embedding_159/StatefulPartitionedCall&^embedding_160/StatefulPartitionedCall&^embedding_161/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_159/StatefulPartitionedCall%embedding_159/StatefulPartitionedCall2N
%embedding_160/StatefulPartitionedCall%embedding_160/StatefulPartitionedCall2N
%embedding_161/StatefulPartitionedCall%embedding_161/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
e
I__inference_flatten_53_layer_call_and_return_conditional_losses_256359423

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_flatten_52_layer_call_and_return_conditional_losses_256359197

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Z
?

N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360354

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_80_greaterequal_y
model_52_256360269
model_52_256360271
model_52_256360273
model_52_256360275
model_53_256360278
model_53_256360280
model_53_256360282
model_53_256360284/
+tf_clip_by_value_80_clip_by_value_minimum_y'
#tf_clip_by_value_80_clip_by_value_y
dense_234_256360296
dense_234_256360298
dense_237_256360301
dense_237_256360303
dense_235_256360306
dense_235_256360308
dense_236_256360311
dense_236_256360313
dense_238_256360316
dense_238_256360318
dense_239_256360323
dense_239_256360325
dense_240_256360329
dense_240_256360331
dense_241_256360336
dense_241_256360338
normalize_26_256360343
normalize_26_256360345
dense_242_256360348
dense_242_256360350
identity??!dense_234/StatefulPartitionedCall?!dense_235/StatefulPartitionedCall?!dense_236/StatefulPartitionedCall?!dense_237/StatefulPartitionedCall?!dense_238/StatefulPartitionedCall?!dense_239/StatefulPartitionedCall?!dense_240/StatefulPartitionedCall?!dense_241/StatefulPartitionedCall?!dense_242/StatefulPartitionedCall? model_52/StatefulPartitionedCall? model_53/StatefulPartitionedCall?$normalize_26/StatefulPartitionedCall?
%tf.math.greater_equal_80/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_80/GreaterEqual?
 model_52/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_52_256360269model_52_256360271model_52_256360273model_52_256360275*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563594022"
 model_52/StatefulPartitionedCall?
 model_53/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_53_256360278model_53_256360280model_53_256360282model_53_256360284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563596282"
 model_53/StatefulPartitionedCall?
)tf.clip_by_value_80/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_80/clip_by_value/Minimum?
!tf.clip_by_value_80/clip_by_valueMaximum-tf.clip_by_value_80/clip_by_value/Minimum:z:0#tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_80/clip_by_value?
tf.cast_80/CastCast)tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_80/Castv
tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_78/concat/axis?
tf.concat_78/concatConcatV2)model_52/StatefulPartitionedCall:output:0)model_53/StatefulPartitionedCall:output:0!tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_78/concat
tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_79/concat/axis?
tf.concat_79/concatConcatV2%tf.clip_by_value_80/clip_by_value:z:0tf.cast_80/Cast:y:0!tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_79/concat?
!dense_234/StatefulPartitionedCallStatefulPartitionedCalltf.concat_78/concat:output:0dense_234_256360296dense_234_256360298*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_234_layer_call_and_return_conditional_losses_2563597372#
!dense_234/StatefulPartitionedCall?
!dense_237/StatefulPartitionedCallStatefulPartitionedCalltf.concat_79/concat:output:0dense_237_256360301dense_237_256360303*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_237_layer_call_and_return_conditional_losses_2563597632#
!dense_237/StatefulPartitionedCall?
!dense_235/StatefulPartitionedCallStatefulPartitionedCall*dense_234/StatefulPartitionedCall:output:0dense_235_256360306dense_235_256360308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_235_layer_call_and_return_conditional_losses_2563597902#
!dense_235/StatefulPartitionedCall?
!dense_236/StatefulPartitionedCallStatefulPartitionedCall*dense_235/StatefulPartitionedCall:output:0dense_236_256360311dense_236_256360313*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_236_layer_call_and_return_conditional_losses_2563598172#
!dense_236/StatefulPartitionedCall?
!dense_238/StatefulPartitionedCallStatefulPartitionedCall*dense_237/StatefulPartitionedCall:output:0dense_238_256360316dense_238_256360318*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_238_layer_call_and_return_conditional_losses_2563598432#
!dense_238/StatefulPartitionedCall
tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_80/concat/axis?
tf.concat_80/concatConcatV2*dense_236/StatefulPartitionedCall:output:0*dense_238/StatefulPartitionedCall:output:0!tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_80/concat?
!dense_239/StatefulPartitionedCallStatefulPartitionedCalltf.concat_80/concat:output:0dense_239_256360323dense_239_256360325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_239_layer_call_and_return_conditional_losses_2563598712#
!dense_239/StatefulPartitionedCall?
tf.nn.relu_78/ReluRelu*dense_239/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_78/Relu?
!dense_240/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_78/Relu:activations:0dense_240_256360329dense_240_256360331*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_240_layer_call_and_return_conditional_losses_2563598982#
!dense_240/StatefulPartitionedCall?
tf.__operators__.add_160/AddV2AddV2*dense_240/StatefulPartitionedCall:output:0 tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_160/AddV2?
tf.nn.relu_79/ReluRelu"tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_79/Relu?
!dense_241/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_79/Relu:activations:0dense_241_256360336dense_241_256360338*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_241_layer_call_and_return_conditional_losses_2563599262#
!dense_241/StatefulPartitionedCall?
tf.__operators__.add_161/AddV2AddV2*dense_241/StatefulPartitionedCall:output:0 tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
tf.__operators__.add_161/AddV2?
tf.nn.relu_80/ReluRelu"tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_80/Relu?
$normalize_26/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_80/Relu:activations:0normalize_26_256360343normalize_26_256360345*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_normalize_26_layer_call_and_return_conditional_losses_2563599612&
$normalize_26/StatefulPartitionedCall?
!dense_242/StatefulPartitionedCallStatefulPartitionedCall-normalize_26/StatefulPartitionedCall:output:0dense_242_256360348dense_242_256360350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_242_layer_call_and_return_conditional_losses_2563599872#
!dense_242/StatefulPartitionedCall?
IdentityIdentity*dense_242/StatefulPartitionedCall:output:0"^dense_234/StatefulPartitionedCall"^dense_235/StatefulPartitionedCall"^dense_236/StatefulPartitionedCall"^dense_237/StatefulPartitionedCall"^dense_238/StatefulPartitionedCall"^dense_239/StatefulPartitionedCall"^dense_240/StatefulPartitionedCall"^dense_241/StatefulPartitionedCall"^dense_242/StatefulPartitionedCall!^model_52/StatefulPartitionedCall!^model_53/StatefulPartitionedCall%^normalize_26/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2F
!dense_234/StatefulPartitionedCall!dense_234/StatefulPartitionedCall2F
!dense_235/StatefulPartitionedCall!dense_235/StatefulPartitionedCall2F
!dense_236/StatefulPartitionedCall!dense_236/StatefulPartitionedCall2F
!dense_237/StatefulPartitionedCall!dense_237/StatefulPartitionedCall2F
!dense_238/StatefulPartitionedCall!dense_238/StatefulPartitionedCall2F
!dense_239/StatefulPartitionedCall!dense_239/StatefulPartitionedCall2F
!dense_240/StatefulPartitionedCall!dense_240/StatefulPartitionedCall2F
!dense_241/StatefulPartitionedCall!dense_241/StatefulPartitionedCall2F
!dense_242/StatefulPartitionedCall!dense_242/StatefulPartitionedCall2D
 model_52/StatefulPartitionedCall model_52/StatefulPartitionedCall2D
 model_53/StatefulPartitionedCall model_53/StatefulPartitionedCall2L
$normalize_26/StatefulPartitionedCall$normalize_26/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?.
?
G__inference_model_52_layer_call_and_return_conditional_losses_256359322
input_53+
'tf_math_greater_equal_78_greaterequal_y
embedding_158_256359304
embedding_156_256359307
embedding_157_256359312
identity??%embedding_156/StatefulPartitionedCall?%embedding_157/StatefulPartitionedCall?%embedding_158/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCallinput_53*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_52_layer_call_and_return_conditional_losses_2563591972
flatten_52/PartitionedCall?
+tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_78/clip_by_value/Minimum/y?
)tf.clip_by_value_78/clip_by_value/MinimumMinimum#flatten_52/PartitionedCall:output:04tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_78/clip_by_value/Minimum?
#tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_78/clip_by_value/y?
!tf.clip_by_value_78/clip_by_valueMaximum-tf.clip_by_value_78/clip_by_value/Minimum:z:0,tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_78/clip_by_value?
$tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_52/FloorDiv/y?
"tf.compat.v1.floor_div_52/FloorDivFloorDiv%tf.clip_by_value_78/clip_by_value:z:0-tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_52/FloorDiv?
%tf.math.greater_equal_78/GreaterEqualGreaterEqual#flatten_52/PartitionedCall:output:0'tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_78/GreaterEqual?
tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_52/FloorMod/y?
tf.math.floormod_52/FloorModFloorMod%tf.clip_by_value_78/clip_by_value:z:0'tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_52/FloorMod?
%embedding_158/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_78/clip_by_value:z:0embedding_158_256359304*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_158_layer_call_and_return_conditional_losses_2563592252'
%embedding_158/StatefulPartitionedCall?
%embedding_156/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_52/FloorDiv:z:0embedding_156_256359307*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_156_layer_call_and_return_conditional_losses_2563592472'
%embedding_156/StatefulPartitionedCall?
tf.cast_78/CastCast)tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_78/Cast?
tf.__operators__.add_156/AddV2AddV2.embedding_158/StatefulPartitionedCall:output:0.embedding_156/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_156/AddV2?
%embedding_157/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_52/FloorMod:z:0embedding_157_256359312*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_157_layer_call_and_return_conditional_losses_2563592712'
%embedding_157/StatefulPartitionedCall?
tf.__operators__.add_157/AddV2AddV2"tf.__operators__.add_156/AddV2:z:0.embedding_157/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_157/AddV2?
 tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_52/ExpandDims/dim?
tf.expand_dims_52/ExpandDims
ExpandDimstf.cast_78/Cast:y:0)tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_52/ExpandDims?
tf.math.multiply_52/MulMul"tf.__operators__.add_157/AddV2:z:0%tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_52/Mul?
+tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_52/Sum/reduction_indices?
tf.math.reduce_sum_52/SumSumtf.math.multiply_52/Mul:z:04tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_52/Sum?
IdentityIdentity"tf.math.reduce_sum_52/Sum:output:0&^embedding_156/StatefulPartitionedCall&^embedding_157/StatefulPartitionedCall&^embedding_158/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_156/StatefulPartitionedCall%embedding_156/StatefulPartitionedCall2N
%embedding_157/StatefulPartitionedCall%embedding_157/StatefulPartitionedCall2N
%embedding_158/StatefulPartitionedCall%embedding_158/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_53:

_output_shapes
: 
?
?
-__inference_dense_234_layer_call_fn_256361216

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_234_layer_call_and_return_conditional_losses_2563597372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_237_layer_call_and_return_conditional_losses_256359763

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_234_layer_call_and_return_conditional_losses_256361207

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_156_layer_call_and_return_conditional_losses_256361434

inputs
embedding_lookup_256361428
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256361428Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256361428*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256361428*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dense_242_layer_call_fn_256361396

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_242_layer_call_and_return_conditional_losses_2563599872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
G__inference_model_52_layer_call_and_return_conditional_losses_256359290
input_53+
'tf_math_greater_equal_78_greaterequal_y
embedding_158_256359234
embedding_156_256359256
embedding_157_256359280
identity??%embedding_156/StatefulPartitionedCall?%embedding_157/StatefulPartitionedCall?%embedding_158/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCallinput_53*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_52_layer_call_and_return_conditional_losses_2563591972
flatten_52/PartitionedCall?
+tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_78/clip_by_value/Minimum/y?
)tf.clip_by_value_78/clip_by_value/MinimumMinimum#flatten_52/PartitionedCall:output:04tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_78/clip_by_value/Minimum?
#tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_78/clip_by_value/y?
!tf.clip_by_value_78/clip_by_valueMaximum-tf.clip_by_value_78/clip_by_value/Minimum:z:0,tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_78/clip_by_value?
$tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_52/FloorDiv/y?
"tf.compat.v1.floor_div_52/FloorDivFloorDiv%tf.clip_by_value_78/clip_by_value:z:0-tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_52/FloorDiv?
%tf.math.greater_equal_78/GreaterEqualGreaterEqual#flatten_52/PartitionedCall:output:0'tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_78/GreaterEqual?
tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_52/FloorMod/y?
tf.math.floormod_52/FloorModFloorMod%tf.clip_by_value_78/clip_by_value:z:0'tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_52/FloorMod?
%embedding_158/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_78/clip_by_value:z:0embedding_158_256359234*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_158_layer_call_and_return_conditional_losses_2563592252'
%embedding_158/StatefulPartitionedCall?
%embedding_156/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_52/FloorDiv:z:0embedding_156_256359256*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_156_layer_call_and_return_conditional_losses_2563592472'
%embedding_156/StatefulPartitionedCall?
tf.cast_78/CastCast)tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_78/Cast?
tf.__operators__.add_156/AddV2AddV2.embedding_158/StatefulPartitionedCall:output:0.embedding_156/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_156/AddV2?
%embedding_157/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_52/FloorMod:z:0embedding_157_256359280*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_157_layer_call_and_return_conditional_losses_2563592712'
%embedding_157/StatefulPartitionedCall?
tf.__operators__.add_157/AddV2AddV2"tf.__operators__.add_156/AddV2:z:0.embedding_157/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_157/AddV2?
 tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_52/ExpandDims/dim?
tf.expand_dims_52/ExpandDims
ExpandDimstf.cast_78/Cast:y:0)tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_52/ExpandDims?
tf.math.multiply_52/MulMul"tf.__operators__.add_157/AddV2:z:0%tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_52/Mul?
+tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_52/Sum/reduction_indices?
tf.math.reduce_sum_52/SumSumtf.math.multiply_52/Mul:z:04tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_52/Sum?
IdentityIdentity"tf.math.reduce_sum_52/Sum:output:0&^embedding_156/StatefulPartitionedCall&^embedding_157/StatefulPartitionedCall&^embedding_158/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2N
%embedding_156/StatefulPartitionedCall%embedding_156/StatefulPartitionedCall2N
%embedding_157/StatefulPartitionedCall%embedding_157/StatefulPartitionedCall2N
%embedding_158/StatefulPartitionedCall%embedding_158/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_53:

_output_shapes
: 
?
w
1__inference_embedding_159_layer_call_fn_256361503

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_159_layer_call_and_return_conditional_losses_2563594732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_flatten_53_layer_call_and_return_conditional_losses_256361464

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
G__inference_model_53_layer_call_and_return_conditional_losses_256361170

inputs+
'tf_math_greater_equal_79_greaterequal_y,
(embedding_161_embedding_lookup_256361144,
(embedding_159_embedding_lookup_256361150,
(embedding_160_embedding_lookup_256361158
identity??embedding_159/embedding_lookup?embedding_160/embedding_lookup?embedding_161/embedding_lookupu
flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_53/Const?
flatten_53/ReshapeReshapeinputsflatten_53/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_53/Reshape?
+tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_79/clip_by_value/Minimum/y?
)tf.clip_by_value_79/clip_by_value/MinimumMinimumflatten_53/Reshape:output:04tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_79/clip_by_value/Minimum?
#tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_79/clip_by_value/y?
!tf.clip_by_value_79/clip_by_valueMaximum-tf.clip_by_value_79/clip_by_value/Minimum:z:0,tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_79/clip_by_value?
$tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_53/FloorDiv/y?
"tf.compat.v1.floor_div_53/FloorDivFloorDiv%tf.clip_by_value_79/clip_by_value:z:0-tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_53/FloorDiv?
%tf.math.greater_equal_79/GreaterEqualGreaterEqualflatten_53/Reshape:output:0'tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_79/GreaterEqual?
tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_53/FloorMod/y?
tf.math.floormod_53/FloorModFloorMod%tf.clip_by_value_79/clip_by_value:z:0'tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_53/FloorMod?
embedding_161/CastCast%tf.clip_by_value_79/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_161/Cast?
embedding_161/embedding_lookupResourceGather(embedding_161_embedding_lookup_256361144embedding_161/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_161/embedding_lookup/256361144*,
_output_shapes
:??????????*
dtype02 
embedding_161/embedding_lookup?
'embedding_161/embedding_lookup/IdentityIdentity'embedding_161/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_161/embedding_lookup/256361144*,
_output_shapes
:??????????2)
'embedding_161/embedding_lookup/Identity?
)embedding_161/embedding_lookup/Identity_1Identity0embedding_161/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_161/embedding_lookup/Identity_1?
embedding_159/CastCast&tf.compat.v1.floor_div_53/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_159/Cast?
embedding_159/embedding_lookupResourceGather(embedding_159_embedding_lookup_256361150embedding_159/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_159/embedding_lookup/256361150*,
_output_shapes
:??????????*
dtype02 
embedding_159/embedding_lookup?
'embedding_159/embedding_lookup/IdentityIdentity'embedding_159/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_159/embedding_lookup/256361150*,
_output_shapes
:??????????2)
'embedding_159/embedding_lookup/Identity?
)embedding_159/embedding_lookup/Identity_1Identity0embedding_159/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_159/embedding_lookup/Identity_1?
tf.cast_79/CastCast)tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_79/Cast?
tf.__operators__.add_158/AddV2AddV22embedding_161/embedding_lookup/Identity_1:output:02embedding_159/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_158/AddV2?
embedding_160/CastCast tf.math.floormod_53/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_160/Cast?
embedding_160/embedding_lookupResourceGather(embedding_160_embedding_lookup_256361158embedding_160/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@embedding_160/embedding_lookup/256361158*,
_output_shapes
:??????????*
dtype02 
embedding_160/embedding_lookup?
'embedding_160/embedding_lookup/IdentityIdentity'embedding_160/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@embedding_160/embedding_lookup/256361158*,
_output_shapes
:??????????2)
'embedding_160/embedding_lookup/Identity?
)embedding_160/embedding_lookup/Identity_1Identity0embedding_160/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2+
)embedding_160/embedding_lookup/Identity_1?
tf.__operators__.add_159/AddV2AddV2"tf.__operators__.add_158/AddV2:z:02embedding_160/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2 
tf.__operators__.add_159/AddV2?
 tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_53/ExpandDims/dim?
tf.expand_dims_53/ExpandDims
ExpandDimstf.cast_79/Cast:y:0)tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_53/ExpandDims?
tf.math.multiply_53/MulMul"tf.__operators__.add_159/AddV2:z:0%tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_53/Mul?
+tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_53/Sum/reduction_indices?
tf.math.reduce_sum_53/SumSumtf.math.multiply_53/Mul:z:04tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_53/Sum?
IdentityIdentity"tf.math.reduce_sum_53/Sum:output:0^embedding_159/embedding_lookup^embedding_160/embedding_lookup^embedding_161/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2@
embedding_159/embedding_lookupembedding_159/embedding_lookup2@
embedding_160/embedding_lookupembedding_160/embedding_lookup2@
embedding_161/embedding_lookupembedding_161/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_model_53_layer_call_fn_256359594
input_54
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_54unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563595832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_54:

_output_shapes
: 
?
?
,__inference_model_52_layer_call_fn_256361086

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563594022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
-__inference_dense_241_layer_call_fn_256361351

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_241_layer_call_and_return_conditional_losses_2563599262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
̡
?#
"__inference__traced_save_256361796
file_prefix/
+savev2_dense_234_kernel_read_readvariableop-
)savev2_dense_234_bias_read_readvariableop/
+savev2_dense_235_kernel_read_readvariableop-
)savev2_dense_235_bias_read_readvariableop/
+savev2_dense_237_kernel_read_readvariableop-
)savev2_dense_237_bias_read_readvariableop/
+savev2_dense_236_kernel_read_readvariableop-
)savev2_dense_236_bias_read_readvariableop/
+savev2_dense_238_kernel_read_readvariableop-
)savev2_dense_238_bias_read_readvariableop/
+savev2_dense_239_kernel_read_readvariableop-
)savev2_dense_239_bias_read_readvariableop/
+savev2_dense_240_kernel_read_readvariableop-
)savev2_dense_240_bias_read_readvariableop/
+savev2_dense_241_kernel_read_readvariableop-
)savev2_dense_241_bias_read_readvariableop/
+savev2_dense_242_kernel_read_readvariableop-
)savev2_dense_242_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_embedding_158_embeddings_read_readvariableop7
3savev2_embedding_156_embeddings_read_readvariableop7
3savev2_embedding_157_embeddings_read_readvariableop7
3savev2_embedding_161_embeddings_read_readvariableop7
3savev2_embedding_159_embeddings_read_readvariableop7
3savev2_embedding_160_embeddings_read_readvariableopA
=savev2_normalize_26_normalization_26_mean_read_readvariableopE
Asavev2_normalize_26_normalization_26_variance_read_readvariableopB
>savev2_normalize_26_normalization_26_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_234_kernel_m_read_readvariableop4
0savev2_adam_dense_234_bias_m_read_readvariableop6
2savev2_adam_dense_235_kernel_m_read_readvariableop4
0savev2_adam_dense_235_bias_m_read_readvariableop6
2savev2_adam_dense_237_kernel_m_read_readvariableop4
0savev2_adam_dense_237_bias_m_read_readvariableop6
2savev2_adam_dense_236_kernel_m_read_readvariableop4
0savev2_adam_dense_236_bias_m_read_readvariableop6
2savev2_adam_dense_238_kernel_m_read_readvariableop4
0savev2_adam_dense_238_bias_m_read_readvariableop6
2savev2_adam_dense_239_kernel_m_read_readvariableop4
0savev2_adam_dense_239_bias_m_read_readvariableop6
2savev2_adam_dense_240_kernel_m_read_readvariableop4
0savev2_adam_dense_240_bias_m_read_readvariableop6
2savev2_adam_dense_241_kernel_m_read_readvariableop4
0savev2_adam_dense_241_bias_m_read_readvariableop6
2savev2_adam_dense_242_kernel_m_read_readvariableop4
0savev2_adam_dense_242_bias_m_read_readvariableop>
:savev2_adam_embedding_158_embeddings_m_read_readvariableop>
:savev2_adam_embedding_156_embeddings_m_read_readvariableop>
:savev2_adam_embedding_157_embeddings_m_read_readvariableop>
:savev2_adam_embedding_161_embeddings_m_read_readvariableop>
:savev2_adam_embedding_159_embeddings_m_read_readvariableop>
:savev2_adam_embedding_160_embeddings_m_read_readvariableop6
2savev2_adam_dense_234_kernel_v_read_readvariableop4
0savev2_adam_dense_234_bias_v_read_readvariableop6
2savev2_adam_dense_235_kernel_v_read_readvariableop4
0savev2_adam_dense_235_bias_v_read_readvariableop6
2savev2_adam_dense_237_kernel_v_read_readvariableop4
0savev2_adam_dense_237_bias_v_read_readvariableop6
2savev2_adam_dense_236_kernel_v_read_readvariableop4
0savev2_adam_dense_236_bias_v_read_readvariableop6
2savev2_adam_dense_238_kernel_v_read_readvariableop4
0savev2_adam_dense_238_bias_v_read_readvariableop6
2savev2_adam_dense_239_kernel_v_read_readvariableop4
0savev2_adam_dense_239_bias_v_read_readvariableop6
2savev2_adam_dense_240_kernel_v_read_readvariableop4
0savev2_adam_dense_240_bias_v_read_readvariableop6
2savev2_adam_dense_241_kernel_v_read_readvariableop4
0savev2_adam_dense_241_bias_v_read_readvariableop6
2savev2_adam_dense_242_kernel_v_read_readvariableop4
0savev2_adam_dense_242_bias_v_read_readvariableop>
:savev2_adam_embedding_158_embeddings_v_read_readvariableop>
:savev2_adam_embedding_156_embeddings_v_read_readvariableop>
:savev2_adam_embedding_157_embeddings_v_read_readvariableop>
:savev2_adam_embedding_161_embeddings_v_read_readvariableop>
:savev2_adam_embedding_159_embeddings_v_read_readvariableop>
:savev2_adam_embedding_160_embeddings_v_read_readvariableop
savev2_const_5

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?,
value?,B?,SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_234_kernel_read_readvariableop)savev2_dense_234_bias_read_readvariableop+savev2_dense_235_kernel_read_readvariableop)savev2_dense_235_bias_read_readvariableop+savev2_dense_237_kernel_read_readvariableop)savev2_dense_237_bias_read_readvariableop+savev2_dense_236_kernel_read_readvariableop)savev2_dense_236_bias_read_readvariableop+savev2_dense_238_kernel_read_readvariableop)savev2_dense_238_bias_read_readvariableop+savev2_dense_239_kernel_read_readvariableop)savev2_dense_239_bias_read_readvariableop+savev2_dense_240_kernel_read_readvariableop)savev2_dense_240_bias_read_readvariableop+savev2_dense_241_kernel_read_readvariableop)savev2_dense_241_bias_read_readvariableop+savev2_dense_242_kernel_read_readvariableop)savev2_dense_242_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_embedding_158_embeddings_read_readvariableop3savev2_embedding_156_embeddings_read_readvariableop3savev2_embedding_157_embeddings_read_readvariableop3savev2_embedding_161_embeddings_read_readvariableop3savev2_embedding_159_embeddings_read_readvariableop3savev2_embedding_160_embeddings_read_readvariableop=savev2_normalize_26_normalization_26_mean_read_readvariableopAsavev2_normalize_26_normalization_26_variance_read_readvariableop>savev2_normalize_26_normalization_26_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_234_kernel_m_read_readvariableop0savev2_adam_dense_234_bias_m_read_readvariableop2savev2_adam_dense_235_kernel_m_read_readvariableop0savev2_adam_dense_235_bias_m_read_readvariableop2savev2_adam_dense_237_kernel_m_read_readvariableop0savev2_adam_dense_237_bias_m_read_readvariableop2savev2_adam_dense_236_kernel_m_read_readvariableop0savev2_adam_dense_236_bias_m_read_readvariableop2savev2_adam_dense_238_kernel_m_read_readvariableop0savev2_adam_dense_238_bias_m_read_readvariableop2savev2_adam_dense_239_kernel_m_read_readvariableop0savev2_adam_dense_239_bias_m_read_readvariableop2savev2_adam_dense_240_kernel_m_read_readvariableop0savev2_adam_dense_240_bias_m_read_readvariableop2savev2_adam_dense_241_kernel_m_read_readvariableop0savev2_adam_dense_241_bias_m_read_readvariableop2savev2_adam_dense_242_kernel_m_read_readvariableop0savev2_adam_dense_242_bias_m_read_readvariableop:savev2_adam_embedding_158_embeddings_m_read_readvariableop:savev2_adam_embedding_156_embeddings_m_read_readvariableop:savev2_adam_embedding_157_embeddings_m_read_readvariableop:savev2_adam_embedding_161_embeddings_m_read_readvariableop:savev2_adam_embedding_159_embeddings_m_read_readvariableop:savev2_adam_embedding_160_embeddings_m_read_readvariableop2savev2_adam_dense_234_kernel_v_read_readvariableop0savev2_adam_dense_234_bias_v_read_readvariableop2savev2_adam_dense_235_kernel_v_read_readvariableop0savev2_adam_dense_235_bias_v_read_readvariableop2savev2_adam_dense_237_kernel_v_read_readvariableop0savev2_adam_dense_237_bias_v_read_readvariableop2savev2_adam_dense_236_kernel_v_read_readvariableop0savev2_adam_dense_236_bias_v_read_readvariableop2savev2_adam_dense_238_kernel_v_read_readvariableop0savev2_adam_dense_238_bias_v_read_readvariableop2savev2_adam_dense_239_kernel_v_read_readvariableop0savev2_adam_dense_239_bias_v_read_readvariableop2savev2_adam_dense_240_kernel_v_read_readvariableop0savev2_adam_dense_240_bias_v_read_readvariableop2savev2_adam_dense_241_kernel_v_read_readvariableop0savev2_adam_dense_241_bias_v_read_readvariableop2savev2_adam_dense_242_kernel_v_read_readvariableop0savev2_adam_dense_242_bias_v_read_readvariableop:savev2_adam_embedding_158_embeddings_v_read_readvariableop:savev2_adam_embedding_156_embeddings_v_read_readvariableop:savev2_adam_embedding_157_embeddings_v_read_readvariableop:savev2_adam_embedding_161_embeddings_v_read_readvariableop:savev2_adam_embedding_159_embeddings_v_read_readvariableop:savev2_adam_embedding_160_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?:: : : : : :	4?:	?:	?:	4?:	?:	?:?:?: : : :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?:
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?: 4

_output_shapes
::%5!

_output_shapes
:	4?:%6!

_output_shapes
:	?:%7!

_output_shapes
:	?:%8!

_output_shapes
:	4?:%9!

_output_shapes
:	?:%:!

_output_shapes
:	?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:%?!

_output_shapes
:	?:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:&C"
 
_output_shapes
:
??:!D

_output_shapes	
:?:&E"
 
_output_shapes
:
??:!F

_output_shapes	
:?:&G"
 
_output_shapes
:
??:!H

_output_shapes	
:?:&I"
 
_output_shapes
:
??:!J

_output_shapes	
:?:%K!

_output_shapes
:	?: L

_output_shapes
::%M!

_output_shapes
:	4?:%N!

_output_shapes
:	?:%O!

_output_shapes
:	?:%P!

_output_shapes
:	4?:%Q!

_output_shapes
:	?:%R!

_output_shapes
:	?:S

_output_shapes
: 
?	
?
H__inference_dense_236_layer_call_and_return_conditional_losses_256359817

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_52_layer_call_fn_256359413
input_53
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_53unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563594022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_53:

_output_shapes
: 
?
?
,__inference_model_52_layer_call_fn_256359368
input_53
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_53unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_52_layer_call_and_return_conditional_losses_2563593572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_53:

_output_shapes
: 
?	
?
L__inference_embedding_159_layer_call_and_return_conditional_losses_256359473

inputs
embedding_lookup_256359467
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256359467Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256359467*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256359467*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_241_layer_call_and_return_conditional_losses_256361342

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_flatten_53_layer_call_fn_256361469

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_flatten_53_layer_call_and_return_conditional_losses_2563594232
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_model_53_layer_call_fn_256361196

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563596282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
-__inference_dense_237_layer_call_fn_256361255

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_dense_237_layer_call_and_return_conditional_losses_2563597632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_236_layer_call_and_return_conditional_losses_256361266

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?,
%__inference__traced_restore_256362052
file_prefix%
!assignvariableop_dense_234_kernel%
!assignvariableop_1_dense_234_bias'
#assignvariableop_2_dense_235_kernel%
!assignvariableop_3_dense_235_bias'
#assignvariableop_4_dense_237_kernel%
!assignvariableop_5_dense_237_bias'
#assignvariableop_6_dense_236_kernel%
!assignvariableop_7_dense_236_bias'
#assignvariableop_8_dense_238_kernel%
!assignvariableop_9_dense_238_bias(
$assignvariableop_10_dense_239_kernel&
"assignvariableop_11_dense_239_bias(
$assignvariableop_12_dense_240_kernel&
"assignvariableop_13_dense_240_bias(
$assignvariableop_14_dense_241_kernel&
"assignvariableop_15_dense_241_bias(
$assignvariableop_16_dense_242_kernel&
"assignvariableop_17_dense_242_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate0
,assignvariableop_23_embedding_158_embeddings0
,assignvariableop_24_embedding_156_embeddings0
,assignvariableop_25_embedding_157_embeddings0
,assignvariableop_26_embedding_161_embeddings0
,assignvariableop_27_embedding_159_embeddings0
,assignvariableop_28_embedding_160_embeddings:
6assignvariableop_29_normalize_26_normalization_26_mean>
:assignvariableop_30_normalize_26_normalization_26_variance;
7assignvariableop_31_normalize_26_normalization_26_count
assignvariableop_32_total
assignvariableop_33_count/
+assignvariableop_34_adam_dense_234_kernel_m-
)assignvariableop_35_adam_dense_234_bias_m/
+assignvariableop_36_adam_dense_235_kernel_m-
)assignvariableop_37_adam_dense_235_bias_m/
+assignvariableop_38_adam_dense_237_kernel_m-
)assignvariableop_39_adam_dense_237_bias_m/
+assignvariableop_40_adam_dense_236_kernel_m-
)assignvariableop_41_adam_dense_236_bias_m/
+assignvariableop_42_adam_dense_238_kernel_m-
)assignvariableop_43_adam_dense_238_bias_m/
+assignvariableop_44_adam_dense_239_kernel_m-
)assignvariableop_45_adam_dense_239_bias_m/
+assignvariableop_46_adam_dense_240_kernel_m-
)assignvariableop_47_adam_dense_240_bias_m/
+assignvariableop_48_adam_dense_241_kernel_m-
)assignvariableop_49_adam_dense_241_bias_m/
+assignvariableop_50_adam_dense_242_kernel_m-
)assignvariableop_51_adam_dense_242_bias_m7
3assignvariableop_52_adam_embedding_158_embeddings_m7
3assignvariableop_53_adam_embedding_156_embeddings_m7
3assignvariableop_54_adam_embedding_157_embeddings_m7
3assignvariableop_55_adam_embedding_161_embeddings_m7
3assignvariableop_56_adam_embedding_159_embeddings_m7
3assignvariableop_57_adam_embedding_160_embeddings_m/
+assignvariableop_58_adam_dense_234_kernel_v-
)assignvariableop_59_adam_dense_234_bias_v/
+assignvariableop_60_adam_dense_235_kernel_v-
)assignvariableop_61_adam_dense_235_bias_v/
+assignvariableop_62_adam_dense_237_kernel_v-
)assignvariableop_63_adam_dense_237_bias_v/
+assignvariableop_64_adam_dense_236_kernel_v-
)assignvariableop_65_adam_dense_236_bias_v/
+assignvariableop_66_adam_dense_238_kernel_v-
)assignvariableop_67_adam_dense_238_bias_v/
+assignvariableop_68_adam_dense_239_kernel_v-
)assignvariableop_69_adam_dense_239_bias_v/
+assignvariableop_70_adam_dense_240_kernel_v-
)assignvariableop_71_adam_dense_240_bias_v/
+assignvariableop_72_adam_dense_241_kernel_v-
)assignvariableop_73_adam_dense_241_bias_v/
+assignvariableop_74_adam_dense_242_kernel_v-
)assignvariableop_75_adam_dense_242_bias_v7
3assignvariableop_76_adam_embedding_158_embeddings_v7
3assignvariableop_77_adam_embedding_156_embeddings_v7
3assignvariableop_78_adam_embedding_157_embeddings_v7
3assignvariableop_79_adam_embedding_161_embeddings_v7
3assignvariableop_80_adam_embedding_159_embeddings_v7
3assignvariableop_81_adam_embedding_160_embeddings_v
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?,
value?,B?,SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_234_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_234_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_235_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_235_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_237_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_237_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_236_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_236_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_238_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_238_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_239_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_239_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_240_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_240_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dense_241_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_241_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_dense_242_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_242_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_embedding_158_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_embedding_156_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_embedding_157_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp,assignvariableop_26_embedding_161_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_embedding_159_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_embedding_160_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_normalize_26_normalization_26_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp:assignvariableop_30_normalize_26_normalization_26_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_normalize_26_normalization_26_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_dense_234_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_234_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_dense_235_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_235_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_dense_237_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_237_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_dense_236_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_236_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_dense_238_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_238_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_dense_239_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_239_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_dense_240_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_240_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dense_241_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_241_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_dense_242_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_242_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_embedding_158_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp3assignvariableop_53_adam_embedding_156_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_embedding_157_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp3assignvariableop_55_adam_embedding_161_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_embedding_159_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp3assignvariableop_57_adam_embedding_160_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp+assignvariableop_58_adam_dense_234_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_234_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp+assignvariableop_60_adam_dense_235_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_235_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_dense_237_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_dense_237_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp+assignvariableop_64_adam_dense_236_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_dense_236_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_dense_238_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_dense_238_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_dense_239_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_239_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_dense_240_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_dense_240_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_dense_241_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_dense_241_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_dense_242_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_242_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp3assignvariableop_76_adam_embedding_158_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp3assignvariableop_77_adam_embedding_156_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp3assignvariableop_78_adam_embedding_157_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp3assignvariableop_79_adam_embedding_161_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_adam_embedding_159_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp3assignvariableop_81_adam_embedding_160_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82?
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
H__inference_dense_235_layer_call_and_return_conditional_losses_256359790

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_normalize_26_layer_call_and_return_conditional_losses_256359961
x4
0normalization_26_reshape_readvariableop_resource6
2normalization_26_reshape_1_readvariableop_resource
identity??'normalization_26/Reshape/ReadVariableOp?)normalization_26/Reshape_1/ReadVariableOp?
'normalization_26/Reshape/ReadVariableOpReadVariableOp0normalization_26_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'normalization_26/Reshape/ReadVariableOp?
normalization_26/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_26/Reshape/shape?
normalization_26/ReshapeReshape/normalization_26/Reshape/ReadVariableOp:value:0'normalization_26/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_26/Reshape?
)normalization_26/Reshape_1/ReadVariableOpReadVariableOp2normalization_26_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)normalization_26/Reshape_1/ReadVariableOp?
 normalization_26/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_26/Reshape_1/shape?
normalization_26/Reshape_1Reshape1normalization_26/Reshape_1/ReadVariableOp:value:0)normalization_26/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_26/Reshape_1?
normalization_26/subSubx!normalization_26/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_26/sub?
normalization_26/SqrtSqrt#normalization_26/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_26/Sqrt}
normalization_26/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_26/Maximum/y?
normalization_26/MaximumMaximumnormalization_26/Sqrt:y:0#normalization_26/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_26/Maximum?
normalization_26/truedivRealDivnormalization_26/sub:z:0normalization_26/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_26/truediv?
IdentityIdentitynormalization_26/truediv:z:0(^normalization_26/Reshape/ReadVariableOp*^normalization_26/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2R
'normalization_26/Reshape/ReadVariableOp'normalization_26/Reshape/ReadVariableOp2V
)normalization_26/Reshape_1/ReadVariableOp)normalization_26/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
w
1__inference_embedding_157_layer_call_fn_256361458

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_157_layer_call_and_return_conditional_losses_2563592712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_157_layer_call_and_return_conditional_losses_256361451

inputs
embedding_lookup_256361445
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256361445Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256361445*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256361445*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_242_layer_call_and_return_conditional_losses_256361387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
w
1__inference_embedding_161_layer_call_fn_256361486

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_161_layer_call_and_return_conditional_losses_2563594512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
$__inference__wrapped_model_256359187

cards0

cards1
bets;
7custom_model_26_tf_math_greater_equal_80_greaterequal_yD
@custom_model_26_model_52_tf_math_greater_equal_78_greaterequal_yE
Acustom_model_26_model_52_embedding_158_embedding_lookup_256359037E
Acustom_model_26_model_52_embedding_156_embedding_lookup_256359043E
Acustom_model_26_model_52_embedding_157_embedding_lookup_256359051D
@custom_model_26_model_53_tf_math_greater_equal_79_greaterequal_yE
Acustom_model_26_model_53_embedding_161_embedding_lookup_256359075E
Acustom_model_26_model_53_embedding_159_embedding_lookup_256359081E
Acustom_model_26_model_53_embedding_160_embedding_lookup_256359089?
;custom_model_26_tf_clip_by_value_80_clip_by_value_minimum_y7
3custom_model_26_tf_clip_by_value_80_clip_by_value_y<
8custom_model_26_dense_234_matmul_readvariableop_resource=
9custom_model_26_dense_234_biasadd_readvariableop_resource<
8custom_model_26_dense_237_matmul_readvariableop_resource=
9custom_model_26_dense_237_biasadd_readvariableop_resource<
8custom_model_26_dense_235_matmul_readvariableop_resource=
9custom_model_26_dense_235_biasadd_readvariableop_resource<
8custom_model_26_dense_236_matmul_readvariableop_resource=
9custom_model_26_dense_236_biasadd_readvariableop_resource<
8custom_model_26_dense_238_matmul_readvariableop_resource=
9custom_model_26_dense_238_biasadd_readvariableop_resource<
8custom_model_26_dense_239_matmul_readvariableop_resource=
9custom_model_26_dense_239_biasadd_readvariableop_resource<
8custom_model_26_dense_240_matmul_readvariableop_resource=
9custom_model_26_dense_240_biasadd_readvariableop_resource<
8custom_model_26_dense_241_matmul_readvariableop_resource=
9custom_model_26_dense_241_biasadd_readvariableop_resourceQ
Mcustom_model_26_normalize_26_normalization_26_reshape_readvariableop_resourceS
Ocustom_model_26_normalize_26_normalization_26_reshape_1_readvariableop_resource<
8custom_model_26_dense_242_matmul_readvariableop_resource=
9custom_model_26_dense_242_biasadd_readvariableop_resource
identity??0custom_model_26/dense_234/BiasAdd/ReadVariableOp?/custom_model_26/dense_234/MatMul/ReadVariableOp?0custom_model_26/dense_235/BiasAdd/ReadVariableOp?/custom_model_26/dense_235/MatMul/ReadVariableOp?0custom_model_26/dense_236/BiasAdd/ReadVariableOp?/custom_model_26/dense_236/MatMul/ReadVariableOp?0custom_model_26/dense_237/BiasAdd/ReadVariableOp?/custom_model_26/dense_237/MatMul/ReadVariableOp?0custom_model_26/dense_238/BiasAdd/ReadVariableOp?/custom_model_26/dense_238/MatMul/ReadVariableOp?0custom_model_26/dense_239/BiasAdd/ReadVariableOp?/custom_model_26/dense_239/MatMul/ReadVariableOp?0custom_model_26/dense_240/BiasAdd/ReadVariableOp?/custom_model_26/dense_240/MatMul/ReadVariableOp?0custom_model_26/dense_241/BiasAdd/ReadVariableOp?/custom_model_26/dense_241/MatMul/ReadVariableOp?0custom_model_26/dense_242/BiasAdd/ReadVariableOp?/custom_model_26/dense_242/MatMul/ReadVariableOp?7custom_model_26/model_52/embedding_156/embedding_lookup?7custom_model_26/model_52/embedding_157/embedding_lookup?7custom_model_26/model_52/embedding_158/embedding_lookup?7custom_model_26/model_53/embedding_159/embedding_lookup?7custom_model_26/model_53/embedding_160/embedding_lookup?7custom_model_26/model_53/embedding_161/embedding_lookup?Dcustom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOp?Fcustom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOp?
5custom_model_26/tf.math.greater_equal_80/GreaterEqualGreaterEqualbets7custom_model_26_tf_math_greater_equal_80_greaterequal_y*
T0*'
_output_shapes
:?????????
27
5custom_model_26/tf.math.greater_equal_80/GreaterEqual?
)custom_model_26/model_52/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)custom_model_26/model_52/flatten_52/Const?
+custom_model_26/model_52/flatten_52/ReshapeReshapecards02custom_model_26/model_52/flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2-
+custom_model_26/model_52/flatten_52/Reshape?
Dcustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2F
Dcustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/Minimum/y?
Bcustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/MinimumMinimum4custom_model_26/model_52/flatten_52/Reshape:output:0Mcustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2D
Bcustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/Minimum?
<custom_model_26/model_52/tf.clip_by_value_78/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<custom_model_26/model_52/tf.clip_by_value_78/clip_by_value/y?
:custom_model_26/model_52/tf.clip_by_value_78/clip_by_valueMaximumFcustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/Minimum:z:0Ecustom_model_26/model_52/tf.clip_by_value_78/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2<
:custom_model_26/model_52/tf.clip_by_value_78/clip_by_value?
=custom_model_26/model_52/tf.compat.v1.floor_div_52/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2?
=custom_model_26/model_52/tf.compat.v1.floor_div_52/FloorDiv/y?
;custom_model_26/model_52/tf.compat.v1.floor_div_52/FloorDivFloorDiv>custom_model_26/model_52/tf.clip_by_value_78/clip_by_value:z:0Fcustom_model_26/model_52/tf.compat.v1.floor_div_52/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2=
;custom_model_26/model_52/tf.compat.v1.floor_div_52/FloorDiv?
>custom_model_26/model_52/tf.math.greater_equal_78/GreaterEqualGreaterEqual4custom_model_26/model_52/flatten_52/Reshape:output:0@custom_model_26_model_52_tf_math_greater_equal_78_greaterequal_y*
T0*'
_output_shapes
:?????????2@
>custom_model_26/model_52/tf.math.greater_equal_78/GreaterEqual?
7custom_model_26/model_52/tf.math.floormod_52/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@29
7custom_model_26/model_52/tf.math.floormod_52/FloorMod/y?
5custom_model_26/model_52/tf.math.floormod_52/FloorModFloorMod>custom_model_26/model_52/tf.clip_by_value_78/clip_by_value:z:0@custom_model_26/model_52/tf.math.floormod_52/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????27
5custom_model_26/model_52/tf.math.floormod_52/FloorMod?
+custom_model_26/model_52/embedding_158/CastCast>custom_model_26/model_52/tf.clip_by_value_78/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2-
+custom_model_26/model_52/embedding_158/Cast?
7custom_model_26/model_52/embedding_158/embedding_lookupResourceGatherAcustom_model_26_model_52_embedding_158_embedding_lookup_256359037/custom_model_26/model_52/embedding_158/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@custom_model_26/model_52/embedding_158/embedding_lookup/256359037*,
_output_shapes
:??????????*
dtype029
7custom_model_26/model_52/embedding_158/embedding_lookup?
@custom_model_26/model_52/embedding_158/embedding_lookup/IdentityIdentity@custom_model_26/model_52/embedding_158/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@custom_model_26/model_52/embedding_158/embedding_lookup/256359037*,
_output_shapes
:??????????2B
@custom_model_26/model_52/embedding_158/embedding_lookup/Identity?
Bcustom_model_26/model_52/embedding_158/embedding_lookup/Identity_1IdentityIcustom_model_26/model_52/embedding_158/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2D
Bcustom_model_26/model_52/embedding_158/embedding_lookup/Identity_1?
+custom_model_26/model_52/embedding_156/CastCast?custom_model_26/model_52/tf.compat.v1.floor_div_52/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2-
+custom_model_26/model_52/embedding_156/Cast?
7custom_model_26/model_52/embedding_156/embedding_lookupResourceGatherAcustom_model_26_model_52_embedding_156_embedding_lookup_256359043/custom_model_26/model_52/embedding_156/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@custom_model_26/model_52/embedding_156/embedding_lookup/256359043*,
_output_shapes
:??????????*
dtype029
7custom_model_26/model_52/embedding_156/embedding_lookup?
@custom_model_26/model_52/embedding_156/embedding_lookup/IdentityIdentity@custom_model_26/model_52/embedding_156/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@custom_model_26/model_52/embedding_156/embedding_lookup/256359043*,
_output_shapes
:??????????2B
@custom_model_26/model_52/embedding_156/embedding_lookup/Identity?
Bcustom_model_26/model_52/embedding_156/embedding_lookup/Identity_1IdentityIcustom_model_26/model_52/embedding_156/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2D
Bcustom_model_26/model_52/embedding_156/embedding_lookup/Identity_1?
(custom_model_26/model_52/tf.cast_78/CastCastBcustom_model_26/model_52/tf.math.greater_equal_78/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2*
(custom_model_26/model_52/tf.cast_78/Cast?
7custom_model_26/model_52/tf.__operators__.add_156/AddV2AddV2Kcustom_model_26/model_52/embedding_158/embedding_lookup/Identity_1:output:0Kcustom_model_26/model_52/embedding_156/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????29
7custom_model_26/model_52/tf.__operators__.add_156/AddV2?
+custom_model_26/model_52/embedding_157/CastCast9custom_model_26/model_52/tf.math.floormod_52/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2-
+custom_model_26/model_52/embedding_157/Cast?
7custom_model_26/model_52/embedding_157/embedding_lookupResourceGatherAcustom_model_26_model_52_embedding_157_embedding_lookup_256359051/custom_model_26/model_52/embedding_157/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@custom_model_26/model_52/embedding_157/embedding_lookup/256359051*,
_output_shapes
:??????????*
dtype029
7custom_model_26/model_52/embedding_157/embedding_lookup?
@custom_model_26/model_52/embedding_157/embedding_lookup/IdentityIdentity@custom_model_26/model_52/embedding_157/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@custom_model_26/model_52/embedding_157/embedding_lookup/256359051*,
_output_shapes
:??????????2B
@custom_model_26/model_52/embedding_157/embedding_lookup/Identity?
Bcustom_model_26/model_52/embedding_157/embedding_lookup/Identity_1IdentityIcustom_model_26/model_52/embedding_157/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2D
Bcustom_model_26/model_52/embedding_157/embedding_lookup/Identity_1?
7custom_model_26/model_52/tf.__operators__.add_157/AddV2AddV2;custom_model_26/model_52/tf.__operators__.add_156/AddV2:z:0Kcustom_model_26/model_52/embedding_157/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????29
7custom_model_26/model_52/tf.__operators__.add_157/AddV2?
9custom_model_26/model_52/tf.expand_dims_52/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2;
9custom_model_26/model_52/tf.expand_dims_52/ExpandDims/dim?
5custom_model_26/model_52/tf.expand_dims_52/ExpandDims
ExpandDims,custom_model_26/model_52/tf.cast_78/Cast:y:0Bcustom_model_26/model_52/tf.expand_dims_52/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????27
5custom_model_26/model_52/tf.expand_dims_52/ExpandDims?
0custom_model_26/model_52/tf.math.multiply_52/MulMul;custom_model_26/model_52/tf.__operators__.add_157/AddV2:z:0>custom_model_26/model_52/tf.expand_dims_52/ExpandDims:output:0*
T0*,
_output_shapes
:??????????22
0custom_model_26/model_52/tf.math.multiply_52/Mul?
Dcustom_model_26/model_52/tf.math.reduce_sum_52/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2F
Dcustom_model_26/model_52/tf.math.reduce_sum_52/Sum/reduction_indices?
2custom_model_26/model_52/tf.math.reduce_sum_52/SumSum4custom_model_26/model_52/tf.math.multiply_52/Mul:z:0Mcustom_model_26/model_52/tf.math.reduce_sum_52/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????24
2custom_model_26/model_52/tf.math.reduce_sum_52/Sum?
)custom_model_26/model_53/flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)custom_model_26/model_53/flatten_53/Const?
+custom_model_26/model_53/flatten_53/ReshapeReshapecards12custom_model_26/model_53/flatten_53/Const:output:0*
T0*'
_output_shapes
:?????????2-
+custom_model_26/model_53/flatten_53/Reshape?
Dcustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2F
Dcustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/Minimum/y?
Bcustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/MinimumMinimum4custom_model_26/model_53/flatten_53/Reshape:output:0Mcustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2D
Bcustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/Minimum?
<custom_model_26/model_53/tf.clip_by_value_79/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<custom_model_26/model_53/tf.clip_by_value_79/clip_by_value/y?
:custom_model_26/model_53/tf.clip_by_value_79/clip_by_valueMaximumFcustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/Minimum:z:0Ecustom_model_26/model_53/tf.clip_by_value_79/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2<
:custom_model_26/model_53/tf.clip_by_value_79/clip_by_value?
=custom_model_26/model_53/tf.compat.v1.floor_div_53/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2?
=custom_model_26/model_53/tf.compat.v1.floor_div_53/FloorDiv/y?
;custom_model_26/model_53/tf.compat.v1.floor_div_53/FloorDivFloorDiv>custom_model_26/model_53/tf.clip_by_value_79/clip_by_value:z:0Fcustom_model_26/model_53/tf.compat.v1.floor_div_53/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2=
;custom_model_26/model_53/tf.compat.v1.floor_div_53/FloorDiv?
>custom_model_26/model_53/tf.math.greater_equal_79/GreaterEqualGreaterEqual4custom_model_26/model_53/flatten_53/Reshape:output:0@custom_model_26_model_53_tf_math_greater_equal_79_greaterequal_y*
T0*'
_output_shapes
:?????????2@
>custom_model_26/model_53/tf.math.greater_equal_79/GreaterEqual?
7custom_model_26/model_53/tf.math.floormod_53/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@29
7custom_model_26/model_53/tf.math.floormod_53/FloorMod/y?
5custom_model_26/model_53/tf.math.floormod_53/FloorModFloorMod>custom_model_26/model_53/tf.clip_by_value_79/clip_by_value:z:0@custom_model_26/model_53/tf.math.floormod_53/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????27
5custom_model_26/model_53/tf.math.floormod_53/FloorMod?
+custom_model_26/model_53/embedding_161/CastCast>custom_model_26/model_53/tf.clip_by_value_79/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2-
+custom_model_26/model_53/embedding_161/Cast?
7custom_model_26/model_53/embedding_161/embedding_lookupResourceGatherAcustom_model_26_model_53_embedding_161_embedding_lookup_256359075/custom_model_26/model_53/embedding_161/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@custom_model_26/model_53/embedding_161/embedding_lookup/256359075*,
_output_shapes
:??????????*
dtype029
7custom_model_26/model_53/embedding_161/embedding_lookup?
@custom_model_26/model_53/embedding_161/embedding_lookup/IdentityIdentity@custom_model_26/model_53/embedding_161/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@custom_model_26/model_53/embedding_161/embedding_lookup/256359075*,
_output_shapes
:??????????2B
@custom_model_26/model_53/embedding_161/embedding_lookup/Identity?
Bcustom_model_26/model_53/embedding_161/embedding_lookup/Identity_1IdentityIcustom_model_26/model_53/embedding_161/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2D
Bcustom_model_26/model_53/embedding_161/embedding_lookup/Identity_1?
+custom_model_26/model_53/embedding_159/CastCast?custom_model_26/model_53/tf.compat.v1.floor_div_53/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2-
+custom_model_26/model_53/embedding_159/Cast?
7custom_model_26/model_53/embedding_159/embedding_lookupResourceGatherAcustom_model_26_model_53_embedding_159_embedding_lookup_256359081/custom_model_26/model_53/embedding_159/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@custom_model_26/model_53/embedding_159/embedding_lookup/256359081*,
_output_shapes
:??????????*
dtype029
7custom_model_26/model_53/embedding_159/embedding_lookup?
@custom_model_26/model_53/embedding_159/embedding_lookup/IdentityIdentity@custom_model_26/model_53/embedding_159/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@custom_model_26/model_53/embedding_159/embedding_lookup/256359081*,
_output_shapes
:??????????2B
@custom_model_26/model_53/embedding_159/embedding_lookup/Identity?
Bcustom_model_26/model_53/embedding_159/embedding_lookup/Identity_1IdentityIcustom_model_26/model_53/embedding_159/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2D
Bcustom_model_26/model_53/embedding_159/embedding_lookup/Identity_1?
(custom_model_26/model_53/tf.cast_79/CastCastBcustom_model_26/model_53/tf.math.greater_equal_79/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2*
(custom_model_26/model_53/tf.cast_79/Cast?
7custom_model_26/model_53/tf.__operators__.add_158/AddV2AddV2Kcustom_model_26/model_53/embedding_161/embedding_lookup/Identity_1:output:0Kcustom_model_26/model_53/embedding_159/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????29
7custom_model_26/model_53/tf.__operators__.add_158/AddV2?
+custom_model_26/model_53/embedding_160/CastCast9custom_model_26/model_53/tf.math.floormod_53/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2-
+custom_model_26/model_53/embedding_160/Cast?
7custom_model_26/model_53/embedding_160/embedding_lookupResourceGatherAcustom_model_26_model_53_embedding_160_embedding_lookup_256359089/custom_model_26/model_53/embedding_160/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@custom_model_26/model_53/embedding_160/embedding_lookup/256359089*,
_output_shapes
:??????????*
dtype029
7custom_model_26/model_53/embedding_160/embedding_lookup?
@custom_model_26/model_53/embedding_160/embedding_lookup/IdentityIdentity@custom_model_26/model_53/embedding_160/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@custom_model_26/model_53/embedding_160/embedding_lookup/256359089*,
_output_shapes
:??????????2B
@custom_model_26/model_53/embedding_160/embedding_lookup/Identity?
Bcustom_model_26/model_53/embedding_160/embedding_lookup/Identity_1IdentityIcustom_model_26/model_53/embedding_160/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2D
Bcustom_model_26/model_53/embedding_160/embedding_lookup/Identity_1?
7custom_model_26/model_53/tf.__operators__.add_159/AddV2AddV2;custom_model_26/model_53/tf.__operators__.add_158/AddV2:z:0Kcustom_model_26/model_53/embedding_160/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????29
7custom_model_26/model_53/tf.__operators__.add_159/AddV2?
9custom_model_26/model_53/tf.expand_dims_53/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2;
9custom_model_26/model_53/tf.expand_dims_53/ExpandDims/dim?
5custom_model_26/model_53/tf.expand_dims_53/ExpandDims
ExpandDims,custom_model_26/model_53/tf.cast_79/Cast:y:0Bcustom_model_26/model_53/tf.expand_dims_53/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????27
5custom_model_26/model_53/tf.expand_dims_53/ExpandDims?
0custom_model_26/model_53/tf.math.multiply_53/MulMul;custom_model_26/model_53/tf.__operators__.add_159/AddV2:z:0>custom_model_26/model_53/tf.expand_dims_53/ExpandDims:output:0*
T0*,
_output_shapes
:??????????22
0custom_model_26/model_53/tf.math.multiply_53/Mul?
Dcustom_model_26/model_53/tf.math.reduce_sum_53/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2F
Dcustom_model_26/model_53/tf.math.reduce_sum_53/Sum/reduction_indices?
2custom_model_26/model_53/tf.math.reduce_sum_53/SumSum4custom_model_26/model_53/tf.math.multiply_53/Mul:z:0Mcustom_model_26/model_53/tf.math.reduce_sum_53/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????24
2custom_model_26/model_53/tf.math.reduce_sum_53/Sum?
9custom_model_26/tf.clip_by_value_80/clip_by_value/MinimumMinimumbets;custom_model_26_tf_clip_by_value_80_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2;
9custom_model_26/tf.clip_by_value_80/clip_by_value/Minimum?
1custom_model_26/tf.clip_by_value_80/clip_by_valueMaximum=custom_model_26/tf.clip_by_value_80/clip_by_value/Minimum:z:03custom_model_26_tf_clip_by_value_80_clip_by_value_y*
T0*'
_output_shapes
:?????????
23
1custom_model_26/tf.clip_by_value_80/clip_by_value?
custom_model_26/tf.cast_80/CastCast9custom_model_26/tf.math.greater_equal_80/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2!
custom_model_26/tf.cast_80/Cast?
(custom_model_26/tf.concat_78/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(custom_model_26/tf.concat_78/concat/axis?
#custom_model_26/tf.concat_78/concatConcatV2;custom_model_26/model_52/tf.math.reduce_sum_52/Sum:output:0;custom_model_26/model_53/tf.math.reduce_sum_53/Sum:output:01custom_model_26/tf.concat_78/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2%
#custom_model_26/tf.concat_78/concat?
(custom_model_26/tf.concat_79/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(custom_model_26/tf.concat_79/concat/axis?
#custom_model_26/tf.concat_79/concatConcatV25custom_model_26/tf.clip_by_value_80/clip_by_value:z:0#custom_model_26/tf.cast_80/Cast:y:01custom_model_26/tf.concat_79/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2%
#custom_model_26/tf.concat_79/concat?
/custom_model_26/dense_234/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_234_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_234/MatMul/ReadVariableOp?
 custom_model_26/dense_234/MatMulMatMul,custom_model_26/tf.concat_78/concat:output:07custom_model_26/dense_234/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_234/MatMul?
0custom_model_26/dense_234/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_234_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_234/BiasAdd/ReadVariableOp?
!custom_model_26/dense_234/BiasAddBiasAdd*custom_model_26/dense_234/MatMul:product:08custom_model_26/dense_234/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_234/BiasAdd?
custom_model_26/dense_234/ReluRelu*custom_model_26/dense_234/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
custom_model_26/dense_234/Relu?
/custom_model_26/dense_237/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_237_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/custom_model_26/dense_237/MatMul/ReadVariableOp?
 custom_model_26/dense_237/MatMulMatMul,custom_model_26/tf.concat_79/concat:output:07custom_model_26/dense_237/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_237/MatMul?
0custom_model_26/dense_237/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_237_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_237/BiasAdd/ReadVariableOp?
!custom_model_26/dense_237/BiasAddBiasAdd*custom_model_26/dense_237/MatMul:product:08custom_model_26/dense_237/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_237/BiasAdd?
/custom_model_26/dense_235/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_235_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_235/MatMul/ReadVariableOp?
 custom_model_26/dense_235/MatMulMatMul,custom_model_26/dense_234/Relu:activations:07custom_model_26/dense_235/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_235/MatMul?
0custom_model_26/dense_235/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_235_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_235/BiasAdd/ReadVariableOp?
!custom_model_26/dense_235/BiasAddBiasAdd*custom_model_26/dense_235/MatMul:product:08custom_model_26/dense_235/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_235/BiasAdd?
custom_model_26/dense_235/ReluRelu*custom_model_26/dense_235/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
custom_model_26/dense_235/Relu?
/custom_model_26/dense_236/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_236_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_236/MatMul/ReadVariableOp?
 custom_model_26/dense_236/MatMulMatMul,custom_model_26/dense_235/Relu:activations:07custom_model_26/dense_236/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_236/MatMul?
0custom_model_26/dense_236/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_236_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_236/BiasAdd/ReadVariableOp?
!custom_model_26/dense_236/BiasAddBiasAdd*custom_model_26/dense_236/MatMul:product:08custom_model_26/dense_236/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_236/BiasAdd?
custom_model_26/dense_236/ReluRelu*custom_model_26/dense_236/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
custom_model_26/dense_236/Relu?
/custom_model_26/dense_238/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_238_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_238/MatMul/ReadVariableOp?
 custom_model_26/dense_238/MatMulMatMul*custom_model_26/dense_237/BiasAdd:output:07custom_model_26/dense_238/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_238/MatMul?
0custom_model_26/dense_238/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_238_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_238/BiasAdd/ReadVariableOp?
!custom_model_26/dense_238/BiasAddBiasAdd*custom_model_26/dense_238/MatMul:product:08custom_model_26/dense_238/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_238/BiasAdd?
(custom_model_26/tf.concat_80/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(custom_model_26/tf.concat_80/concat/axis?
#custom_model_26/tf.concat_80/concatConcatV2,custom_model_26/dense_236/Relu:activations:0*custom_model_26/dense_238/BiasAdd:output:01custom_model_26/tf.concat_80/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2%
#custom_model_26/tf.concat_80/concat?
/custom_model_26/dense_239/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_239_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_239/MatMul/ReadVariableOp?
 custom_model_26/dense_239/MatMulMatMul,custom_model_26/tf.concat_80/concat:output:07custom_model_26/dense_239/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_239/MatMul?
0custom_model_26/dense_239/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_239_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_239/BiasAdd/ReadVariableOp?
!custom_model_26/dense_239/BiasAddBiasAdd*custom_model_26/dense_239/MatMul:product:08custom_model_26/dense_239/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_239/BiasAdd?
"custom_model_26/tf.nn.relu_78/ReluRelu*custom_model_26/dense_239/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2$
"custom_model_26/tf.nn.relu_78/Relu?
/custom_model_26/dense_240/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_240_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_240/MatMul/ReadVariableOp?
 custom_model_26/dense_240/MatMulMatMul0custom_model_26/tf.nn.relu_78/Relu:activations:07custom_model_26/dense_240/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_240/MatMul?
0custom_model_26/dense_240/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_240_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_240/BiasAdd/ReadVariableOp?
!custom_model_26/dense_240/BiasAddBiasAdd*custom_model_26/dense_240/MatMul:product:08custom_model_26/dense_240/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_240/BiasAdd?
.custom_model_26/tf.__operators__.add_160/AddV2AddV2*custom_model_26/dense_240/BiasAdd:output:00custom_model_26/tf.nn.relu_78/Relu:activations:0*
T0*(
_output_shapes
:??????????20
.custom_model_26/tf.__operators__.add_160/AddV2?
"custom_model_26/tf.nn.relu_79/ReluRelu2custom_model_26/tf.__operators__.add_160/AddV2:z:0*
T0*(
_output_shapes
:??????????2$
"custom_model_26/tf.nn.relu_79/Relu?
/custom_model_26/dense_241/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_241_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/custom_model_26/dense_241/MatMul/ReadVariableOp?
 custom_model_26/dense_241/MatMulMatMul0custom_model_26/tf.nn.relu_79/Relu:activations:07custom_model_26/dense_241/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 custom_model_26/dense_241/MatMul?
0custom_model_26/dense_241/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_241_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0custom_model_26/dense_241/BiasAdd/ReadVariableOp?
!custom_model_26/dense_241/BiasAddBiasAdd*custom_model_26/dense_241/MatMul:product:08custom_model_26/dense_241/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!custom_model_26/dense_241/BiasAdd?
.custom_model_26/tf.__operators__.add_161/AddV2AddV2*custom_model_26/dense_241/BiasAdd:output:00custom_model_26/tf.nn.relu_79/Relu:activations:0*
T0*(
_output_shapes
:??????????20
.custom_model_26/tf.__operators__.add_161/AddV2?
"custom_model_26/tf.nn.relu_80/ReluRelu2custom_model_26/tf.__operators__.add_161/AddV2:z:0*
T0*(
_output_shapes
:??????????2$
"custom_model_26/tf.nn.relu_80/Relu?
Dcustom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOpReadVariableOpMcustom_model_26_normalize_26_normalization_26_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dcustom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOp?
;custom_model_26/normalize_26/normalization_26/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2=
;custom_model_26/normalize_26/normalization_26/Reshape/shape?
5custom_model_26/normalize_26/normalization_26/ReshapeReshapeLcustom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOp:value:0Dcustom_model_26/normalize_26/normalization_26/Reshape/shape:output:0*
T0*
_output_shapes
:	?27
5custom_model_26/normalize_26/normalization_26/Reshape?
Fcustom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOpReadVariableOpOcustom_model_26_normalize_26_normalization_26_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fcustom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOp?
=custom_model_26/normalize_26/normalization_26/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2?
=custom_model_26/normalize_26/normalization_26/Reshape_1/shape?
7custom_model_26/normalize_26/normalization_26/Reshape_1ReshapeNcustom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOp:value:0Fcustom_model_26/normalize_26/normalization_26/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?29
7custom_model_26/normalize_26/normalization_26/Reshape_1?
1custom_model_26/normalize_26/normalization_26/subSub0custom_model_26/tf.nn.relu_80/Relu:activations:0>custom_model_26/normalize_26/normalization_26/Reshape:output:0*
T0*(
_output_shapes
:??????????23
1custom_model_26/normalize_26/normalization_26/sub?
2custom_model_26/normalize_26/normalization_26/SqrtSqrt@custom_model_26/normalize_26/normalization_26/Reshape_1:output:0*
T0*
_output_shapes
:	?24
2custom_model_26/normalize_26/normalization_26/Sqrt?
7custom_model_26/normalize_26/normalization_26/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???329
7custom_model_26/normalize_26/normalization_26/Maximum/y?
5custom_model_26/normalize_26/normalization_26/MaximumMaximum6custom_model_26/normalize_26/normalization_26/Sqrt:y:0@custom_model_26/normalize_26/normalization_26/Maximum/y:output:0*
T0*
_output_shapes
:	?27
5custom_model_26/normalize_26/normalization_26/Maximum?
5custom_model_26/normalize_26/normalization_26/truedivRealDiv5custom_model_26/normalize_26/normalization_26/sub:z:09custom_model_26/normalize_26/normalization_26/Maximum:z:0*
T0*(
_output_shapes
:??????????27
5custom_model_26/normalize_26/normalization_26/truediv?
/custom_model_26/dense_242/MatMul/ReadVariableOpReadVariableOp8custom_model_26_dense_242_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/custom_model_26/dense_242/MatMul/ReadVariableOp?
 custom_model_26/dense_242/MatMulMatMul9custom_model_26/normalize_26/normalization_26/truediv:z:07custom_model_26/dense_242/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 custom_model_26/dense_242/MatMul?
0custom_model_26/dense_242/BiasAdd/ReadVariableOpReadVariableOp9custom_model_26_dense_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0custom_model_26/dense_242/BiasAdd/ReadVariableOp?
!custom_model_26/dense_242/BiasAddBiasAdd*custom_model_26/dense_242/MatMul:product:08custom_model_26/dense_242/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!custom_model_26/dense_242/BiasAdd?
IdentityIdentity*custom_model_26/dense_242/BiasAdd:output:01^custom_model_26/dense_234/BiasAdd/ReadVariableOp0^custom_model_26/dense_234/MatMul/ReadVariableOp1^custom_model_26/dense_235/BiasAdd/ReadVariableOp0^custom_model_26/dense_235/MatMul/ReadVariableOp1^custom_model_26/dense_236/BiasAdd/ReadVariableOp0^custom_model_26/dense_236/MatMul/ReadVariableOp1^custom_model_26/dense_237/BiasAdd/ReadVariableOp0^custom_model_26/dense_237/MatMul/ReadVariableOp1^custom_model_26/dense_238/BiasAdd/ReadVariableOp0^custom_model_26/dense_238/MatMul/ReadVariableOp1^custom_model_26/dense_239/BiasAdd/ReadVariableOp0^custom_model_26/dense_239/MatMul/ReadVariableOp1^custom_model_26/dense_240/BiasAdd/ReadVariableOp0^custom_model_26/dense_240/MatMul/ReadVariableOp1^custom_model_26/dense_241/BiasAdd/ReadVariableOp0^custom_model_26/dense_241/MatMul/ReadVariableOp1^custom_model_26/dense_242/BiasAdd/ReadVariableOp0^custom_model_26/dense_242/MatMul/ReadVariableOp8^custom_model_26/model_52/embedding_156/embedding_lookup8^custom_model_26/model_52/embedding_157/embedding_lookup8^custom_model_26/model_52/embedding_158/embedding_lookup8^custom_model_26/model_53/embedding_159/embedding_lookup8^custom_model_26/model_53/embedding_160/embedding_lookup8^custom_model_26/model_53/embedding_161/embedding_lookupE^custom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOpG^custom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2d
0custom_model_26/dense_234/BiasAdd/ReadVariableOp0custom_model_26/dense_234/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_234/MatMul/ReadVariableOp/custom_model_26/dense_234/MatMul/ReadVariableOp2d
0custom_model_26/dense_235/BiasAdd/ReadVariableOp0custom_model_26/dense_235/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_235/MatMul/ReadVariableOp/custom_model_26/dense_235/MatMul/ReadVariableOp2d
0custom_model_26/dense_236/BiasAdd/ReadVariableOp0custom_model_26/dense_236/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_236/MatMul/ReadVariableOp/custom_model_26/dense_236/MatMul/ReadVariableOp2d
0custom_model_26/dense_237/BiasAdd/ReadVariableOp0custom_model_26/dense_237/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_237/MatMul/ReadVariableOp/custom_model_26/dense_237/MatMul/ReadVariableOp2d
0custom_model_26/dense_238/BiasAdd/ReadVariableOp0custom_model_26/dense_238/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_238/MatMul/ReadVariableOp/custom_model_26/dense_238/MatMul/ReadVariableOp2d
0custom_model_26/dense_239/BiasAdd/ReadVariableOp0custom_model_26/dense_239/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_239/MatMul/ReadVariableOp/custom_model_26/dense_239/MatMul/ReadVariableOp2d
0custom_model_26/dense_240/BiasAdd/ReadVariableOp0custom_model_26/dense_240/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_240/MatMul/ReadVariableOp/custom_model_26/dense_240/MatMul/ReadVariableOp2d
0custom_model_26/dense_241/BiasAdd/ReadVariableOp0custom_model_26/dense_241/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_241/MatMul/ReadVariableOp/custom_model_26/dense_241/MatMul/ReadVariableOp2d
0custom_model_26/dense_242/BiasAdd/ReadVariableOp0custom_model_26/dense_242/BiasAdd/ReadVariableOp2b
/custom_model_26/dense_242/MatMul/ReadVariableOp/custom_model_26/dense_242/MatMul/ReadVariableOp2r
7custom_model_26/model_52/embedding_156/embedding_lookup7custom_model_26/model_52/embedding_156/embedding_lookup2r
7custom_model_26/model_52/embedding_157/embedding_lookup7custom_model_26/model_52/embedding_157/embedding_lookup2r
7custom_model_26/model_52/embedding_158/embedding_lookup7custom_model_26/model_52/embedding_158/embedding_lookup2r
7custom_model_26/model_53/embedding_159/embedding_lookup7custom_model_26/model_53/embedding_159/embedding_lookup2r
7custom_model_26/model_53/embedding_160/embedding_lookup7custom_model_26/model_53/embedding_160/embedding_lookup2r
7custom_model_26/model_53/embedding_161/embedding_lookup7custom_model_26/model_53/embedding_161/embedding_lookup2?
Dcustom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOpDcustom_model_26/normalize_26/normalization_26/Reshape/ReadVariableOp2?
Fcustom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOpFcustom_model_26/normalize_26/normalization_26/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_model_53_layer_call_fn_256361183

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_model_53_layer_call_and_return_conditional_losses_2563595832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
H__inference_dense_234_layer_call_and_return_conditional_losses_256359737

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
w
1__inference_embedding_158_layer_call_fn_256361424

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_embedding_158_layer_call_and_return_conditional_losses_2563592252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_embedding_160_layer_call_and_return_conditional_losses_256361513

inputs
embedding_lookup_256361507
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_256361507Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/256361507*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/256361507*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
bets-
serving_default_bets:0?????????

9
cards0/
serving_default_cards0:0?????????
9
cards1/
serving_default_cards1:0?????????=
	dense_2420
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
trainable_variables
	variables
regularization_losses
 	keras_api
!
signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_networkН{"class_name": "CustomModel", "name": "custom_model_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_26", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_52", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}, "name": "input_53", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["input_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_78", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_78", "inbound_nodes": [["flatten_52", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_52", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_158", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_158", "inbound_nodes": [[["tf.clip_by_value_78", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_156", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_156", "inbound_nodes": [[["tf.compat.v1.floor_div_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_52", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_78", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_78", "inbound_nodes": [["flatten_52", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_156", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_156", "inbound_nodes": [["embedding_158", 0, 0, {"y": ["embedding_156", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_157", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_157", "inbound_nodes": [[["tf.math.floormod_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_78", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_78", "inbound_nodes": [["tf.math.greater_equal_78", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_157", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_157", "inbound_nodes": [["tf.__operators__.add_156", 0, 0, {"y": ["embedding_157", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_52", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_52", "inbound_nodes": [["tf.cast_78", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_52", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_52", "inbound_nodes": [["tf.__operators__.add_157", 0, 0, {"y": ["tf.expand_dims_52", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_52", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_52", "inbound_nodes": [["tf.math.multiply_52", 0, 0, {"axis": 1}]]}], "input_layers": [["input_53", 0, 0]], "output_layers": [["tf.math.reduce_sum_52", 0, 0]]}, "name": "model_52", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_53", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_54"}, "name": "input_54", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_53", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_53", "inbound_nodes": [[["input_54", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_79", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_79", "inbound_nodes": [["flatten_53", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_53", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_161", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_161", "inbound_nodes": [[["tf.clip_by_value_79", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_159", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_159", "inbound_nodes": [[["tf.compat.v1.floor_div_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_53", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_79", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_79", "inbound_nodes": [["flatten_53", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_158", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_158", "inbound_nodes": [["embedding_161", 0, 0, {"y": ["embedding_159", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_160", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_160", "inbound_nodes": [[["tf.math.floormod_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_79", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_79", "inbound_nodes": [["tf.math.greater_equal_79", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_159", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_159", "inbound_nodes": [["tf.__operators__.add_158", 0, 0, {"y": ["embedding_160", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_53", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_53", "inbound_nodes": [["tf.cast_79", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_53", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_53", "inbound_nodes": [["tf.__operators__.add_159", 0, 0, {"y": ["tf.expand_dims_53", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_53", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_53", "inbound_nodes": [["tf.math.multiply_53", 0, 0, {"axis": 1}]]}], "input_layers": [["input_54", 0, 0]], "output_layers": [["tf.math.reduce_sum_53", 0, 0]]}, "name": "model_53", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_80", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_80", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_78", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_78", "inbound_nodes": [[["model_52", 1, 0, {"axis": 1}], ["model_53", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_80", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_80", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_80", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_80", "inbound_nodes": [["tf.math.greater_equal_80", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_234", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_234", "inbound_nodes": [[["tf.concat_78", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_79", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_79", "inbound_nodes": [[["tf.clip_by_value_80", 0, 0, {"axis": -1}], ["tf.cast_80", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_235", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_235", "inbound_nodes": [[["dense_234", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_237", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_237", "inbound_nodes": [[["tf.concat_79", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_236", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_236", "inbound_nodes": [[["dense_235", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_238", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_238", "inbound_nodes": [[["dense_237", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_80", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_80", "inbound_nodes": [[["dense_236", 0, 0, {"axis": -1}], ["dense_238", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_239", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_239", "inbound_nodes": [[["tf.concat_80", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_78", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_78", "inbound_nodes": [["dense_239", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_240", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_240", "inbound_nodes": [[["tf.nn.relu_78", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_160", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_160", "inbound_nodes": [["dense_240", 0, 0, {"y": ["tf.nn.relu_78", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_79", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_79", "inbound_nodes": [["tf.__operators__.add_160", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_241", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_241", "inbound_nodes": [[["tf.nn.relu_79", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_161", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_161", "inbound_nodes": [["dense_241", 0, 0, {"y": ["tf.nn.relu_79", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_80", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_80", "inbound_nodes": [["tf.__operators__.add_161", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_26", "trainable": true, "dtype": "float32"}, "name": "normalize_26", "inbound_nodes": [[["tf.nn.relu_80", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_242", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_242", "inbound_nodes": [[["normalize_26", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_242", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_26", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_52", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}, "name": "input_53", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["input_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_78", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_78", "inbound_nodes": [["flatten_52", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_52", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_158", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_158", "inbound_nodes": [[["tf.clip_by_value_78", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_156", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_156", "inbound_nodes": [[["tf.compat.v1.floor_div_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_52", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_78", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_78", "inbound_nodes": [["flatten_52", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_156", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_156", "inbound_nodes": [["embedding_158", 0, 0, {"y": ["embedding_156", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_157", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_157", "inbound_nodes": [[["tf.math.floormod_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_78", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_78", "inbound_nodes": [["tf.math.greater_equal_78", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_157", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_157", "inbound_nodes": [["tf.__operators__.add_156", 0, 0, {"y": ["embedding_157", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_52", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_52", "inbound_nodes": [["tf.cast_78", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_52", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_52", "inbound_nodes": [["tf.__operators__.add_157", 0, 0, {"y": ["tf.expand_dims_52", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_52", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_52", "inbound_nodes": [["tf.math.multiply_52", 0, 0, {"axis": 1}]]}], "input_layers": [["input_53", 0, 0]], "output_layers": [["tf.math.reduce_sum_52", 0, 0]]}, "name": "model_52", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_53", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_54"}, "name": "input_54", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_53", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_53", "inbound_nodes": [[["input_54", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_79", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_79", "inbound_nodes": [["flatten_53", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_53", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_161", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_161", "inbound_nodes": [[["tf.clip_by_value_79", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_159", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_159", "inbound_nodes": [[["tf.compat.v1.floor_div_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_53", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_79", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_79", "inbound_nodes": [["flatten_53", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_158", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_158", "inbound_nodes": [["embedding_161", 0, 0, {"y": ["embedding_159", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_160", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_160", "inbound_nodes": [[["tf.math.floormod_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_79", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_79", "inbound_nodes": [["tf.math.greater_equal_79", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_159", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_159", "inbound_nodes": [["tf.__operators__.add_158", 0, 0, {"y": ["embedding_160", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_53", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_53", "inbound_nodes": [["tf.cast_79", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_53", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_53", "inbound_nodes": [["tf.__operators__.add_159", 0, 0, {"y": ["tf.expand_dims_53", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_53", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_53", "inbound_nodes": [["tf.math.multiply_53", 0, 0, {"axis": 1}]]}], "input_layers": [["input_54", 0, 0]], "output_layers": [["tf.math.reduce_sum_53", 0, 0]]}, "name": "model_53", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_80", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_80", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_78", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_78", "inbound_nodes": [[["model_52", 1, 0, {"axis": 1}], ["model_53", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_80", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_80", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_80", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_80", "inbound_nodes": [["tf.math.greater_equal_80", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_234", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_234", "inbound_nodes": [[["tf.concat_78", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_79", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_79", "inbound_nodes": [[["tf.clip_by_value_80", 0, 0, {"axis": -1}], ["tf.cast_80", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_235", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_235", "inbound_nodes": [[["dense_234", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_237", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_237", "inbound_nodes": [[["tf.concat_79", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_236", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_236", "inbound_nodes": [[["dense_235", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_238", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_238", "inbound_nodes": [[["dense_237", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_80", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_80", "inbound_nodes": [[["dense_236", 0, 0, {"axis": -1}], ["dense_238", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_239", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_239", "inbound_nodes": [[["tf.concat_80", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_78", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_78", "inbound_nodes": [["dense_239", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_240", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_240", "inbound_nodes": [[["tf.nn.relu_78", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_160", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_160", "inbound_nodes": [["dense_240", 0, 0, {"y": ["tf.nn.relu_78", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_79", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_79", "inbound_nodes": [["tf.__operators__.add_160", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_241", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_241", "inbound_nodes": [[["tf.nn.relu_79", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_161", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_161", "inbound_nodes": [["dense_241", 0, 0, {"y": ["tf.nn.relu_79", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_80", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_80", "inbound_nodes": [["tf.__operators__.add_161", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_26", "trainable": true, "dtype": "float32"}, "name": "normalize_26", "inbound_nodes": [[["tf.nn.relu_80", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_242", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_242", "inbound_nodes": [[["normalize_26", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_242", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001538461510790512, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?R
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?O
_tf_keras_network?N{"class_name": "Functional", "name": "model_52", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_52", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}, "name": "input_53", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["input_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_78", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_78", "inbound_nodes": [["flatten_52", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_52", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_158", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_158", "inbound_nodes": [[["tf.clip_by_value_78", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_156", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_156", "inbound_nodes": [[["tf.compat.v1.floor_div_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_52", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_78", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_78", "inbound_nodes": [["flatten_52", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_156", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_156", "inbound_nodes": [["embedding_158", 0, 0, {"y": ["embedding_156", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_157", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_157", "inbound_nodes": [[["tf.math.floormod_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_78", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_78", "inbound_nodes": [["tf.math.greater_equal_78", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_157", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_157", "inbound_nodes": [["tf.__operators__.add_156", 0, 0, {"y": ["embedding_157", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_52", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_52", "inbound_nodes": [["tf.cast_78", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_52", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_52", "inbound_nodes": [["tf.__operators__.add_157", 0, 0, {"y": ["tf.expand_dims_52", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_52", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_52", "inbound_nodes": [["tf.math.multiply_52", 0, 0, {"axis": 1}]]}], "input_layers": [["input_53", 0, 0]], "output_layers": [["tf.math.reduce_sum_52", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_52", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}, "name": "input_53", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["input_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_78", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_78", "inbound_nodes": [["flatten_52", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_52", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_158", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_158", "inbound_nodes": [[["tf.clip_by_value_78", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_156", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_156", "inbound_nodes": [[["tf.compat.v1.floor_div_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_52", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_52", "inbound_nodes": [["tf.clip_by_value_78", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_78", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_78", "inbound_nodes": [["flatten_52", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_156", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_156", "inbound_nodes": [["embedding_158", 0, 0, {"y": ["embedding_156", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_157", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_157", "inbound_nodes": [[["tf.math.floormod_52", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_78", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_78", "inbound_nodes": [["tf.math.greater_equal_78", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_157", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_157", "inbound_nodes": [["tf.__operators__.add_156", 0, 0, {"y": ["embedding_157", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_52", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_52", "inbound_nodes": [["tf.cast_78", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_52", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_52", "inbound_nodes": [["tf.__operators__.add_157", 0, 0, {"y": ["tf.expand_dims_52", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_52", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_52", "inbound_nodes": [["tf.math.multiply_52", 0, 0, {"axis": 1}]]}], "input_layers": [["input_53", 0, 0]], "output_layers": [["tf.math.reduce_sum_52", 0, 0]]}}}
?R
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?O
_tf_keras_network?N{"class_name": "Functional", "name": "model_53", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_53", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_54"}, "name": "input_54", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_53", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_53", "inbound_nodes": [[["input_54", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_79", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_79", "inbound_nodes": [["flatten_53", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_53", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_161", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_161", "inbound_nodes": [[["tf.clip_by_value_79", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_159", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_159", "inbound_nodes": [[["tf.compat.v1.floor_div_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_53", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_79", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_79", "inbound_nodes": [["flatten_53", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_158", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_158", "inbound_nodes": [["embedding_161", 0, 0, {"y": ["embedding_159", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_160", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_160", "inbound_nodes": [[["tf.math.floormod_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_79", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_79", "inbound_nodes": [["tf.math.greater_equal_79", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_159", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_159", "inbound_nodes": [["tf.__operators__.add_158", 0, 0, {"y": ["embedding_160", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_53", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_53", "inbound_nodes": [["tf.cast_79", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_53", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_53", "inbound_nodes": [["tf.__operators__.add_159", 0, 0, {"y": ["tf.expand_dims_53", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_53", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_53", "inbound_nodes": [["tf.math.multiply_53", 0, 0, {"axis": 1}]]}], "input_layers": [["input_54", 0, 0]], "output_layers": [["tf.math.reduce_sum_53", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_53", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_54"}, "name": "input_54", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_53", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_53", "inbound_nodes": [[["input_54", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_79", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_79", "inbound_nodes": [["flatten_53", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_53", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_161", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_161", "inbound_nodes": [[["tf.clip_by_value_79", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_159", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_159", "inbound_nodes": [[["tf.compat.v1.floor_div_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_53", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_53", "inbound_nodes": [["tf.clip_by_value_79", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_79", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_79", "inbound_nodes": [["flatten_53", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_158", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_158", "inbound_nodes": [["embedding_161", 0, 0, {"y": ["embedding_159", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_160", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_160", "inbound_nodes": [[["tf.math.floormod_53", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_79", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_79", "inbound_nodes": [["tf.math.greater_equal_79", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_159", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_159", "inbound_nodes": [["tf.__operators__.add_158", 0, 0, {"y": ["embedding_160", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_53", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_53", "inbound_nodes": [["tf.cast_79", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_53", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_53", "inbound_nodes": [["tf.__operators__.add_159", 0, 0, {"y": ["tf.expand_dims_53", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_53", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_53", "inbound_nodes": [["tf.math.multiply_53", 0, 0, {"axis": 1}]]}], "input_layers": [["input_54", 0, 0]], "output_layers": [["tf.math.reduce_sum_53", 0, 0]]}}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_80", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_78", "trainable": true, "dtype": "float32", "function": "concat"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_80", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_80", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_234", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_234", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_79", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_235", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_235", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_237", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_237", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

_kernel
`bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_236", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_236", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ekernel
fbias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_238", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_238", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
k	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_80", "trainable": true, "dtype": "float32", "function": "concat"}}
?

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_239", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_239", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
r	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_78", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

skernel
tbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_240", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_240", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_160", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_160", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_79", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_241", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_241", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_161", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_161", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_80", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_26", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_242", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_242", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_53", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_53"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_52", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_78", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_52", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_158", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_158", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_156", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_156", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_52", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_78", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_156", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_156", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_157", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_157", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_78", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_157", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_157", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_52", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_52", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_52", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
1trainable_variables
 ?layer_regularization_losses
2	variables
3regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_54", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_54"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_53", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_79", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_53", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_161", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_161", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_159", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_159", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_53", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_79", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_158", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_158", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_160", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_160", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_79", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_159", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_159", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_53", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_53", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_53", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Dtrainable_variables
 ?layer_regularization_losses
E	variables
Fregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
$:"
??2dense_234/kernel
:?2dense_234/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Ntrainable_variables
 ?layer_regularization_losses
O	variables
Pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
$:"
??2dense_235/kernel
:?2dense_235/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Utrainable_variables
 ?layer_regularization_losses
V	variables
Wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?2dense_237/kernel
:?2dense_237/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
[trainable_variables
 ?layer_regularization_losses
\	variables
]regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_236/kernel
:?2dense_236/bias
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
atrainable_variables
 ?layer_regularization_losses
b	variables
cregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_238/kernel
:?2dense_238/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
gtrainable_variables
 ?layer_regularization_losses
h	variables
iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
$:"
??2dense_239/kernel
:?2dense_239/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
ntrainable_variables
 ?layer_regularization_losses
o	variables
pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
$:"
??2dense_240/kernel
:?2dense_240/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
utrainable_variables
 ?layer_regularization_losses
v	variables
wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
$:"
??2dense_241/kernel
:?2dense_241/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
}trainable_variables
 ?layer_regularization_losses
~	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_26", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?2dense_242/kernel
:2dense_242/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)	4?2embedding_158/embeddings
+:)	?2embedding_156/embeddings
+:)	?2embedding_157/embeddings
+:)	4?2embedding_161/embeddings
+:)	?2embedding_159/embeddings
+:)	?2embedding_160/embeddings
/:-?2"normalize_26/normalization_26/mean
3:1?2&normalize_26/normalization_26/variance
+:)	 2#normalize_26/normalization_26/count
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
F
	?mean
?variance

?count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
):'
??2Adam/dense_234/kernel/m
": ?2Adam/dense_234/bias/m
):'
??2Adam/dense_235/kernel/m
": ?2Adam/dense_235/bias/m
(:&	?2Adam/dense_237/kernel/m
": ?2Adam/dense_237/bias/m
):'
??2Adam/dense_236/kernel/m
": ?2Adam/dense_236/bias/m
):'
??2Adam/dense_238/kernel/m
": ?2Adam/dense_238/bias/m
):'
??2Adam/dense_239/kernel/m
": ?2Adam/dense_239/bias/m
):'
??2Adam/dense_240/kernel/m
": ?2Adam/dense_240/bias/m
):'
??2Adam/dense_241/kernel/m
": ?2Adam/dense_241/bias/m
(:&	?2Adam/dense_242/kernel/m
!:2Adam/dense_242/bias/m
0:.	4?2Adam/embedding_158/embeddings/m
0:.	?2Adam/embedding_156/embeddings/m
0:.	?2Adam/embedding_157/embeddings/m
0:.	4?2Adam/embedding_161/embeddings/m
0:.	?2Adam/embedding_159/embeddings/m
0:.	?2Adam/embedding_160/embeddings/m
):'
??2Adam/dense_234/kernel/v
": ?2Adam/dense_234/bias/v
):'
??2Adam/dense_235/kernel/v
": ?2Adam/dense_235/bias/v
(:&	?2Adam/dense_237/kernel/v
": ?2Adam/dense_237/bias/v
):'
??2Adam/dense_236/kernel/v
": ?2Adam/dense_236/bias/v
):'
??2Adam/dense_238/kernel/v
": ?2Adam/dense_238/bias/v
):'
??2Adam/dense_239/kernel/v
": ?2Adam/dense_239/bias/v
):'
??2Adam/dense_240/kernel/v
": ?2Adam/dense_240/bias/v
):'
??2Adam/dense_241/kernel/v
": ?2Adam/dense_241/bias/v
(:&	?2Adam/dense_242/kernel/v
!:2Adam/dense_242/bias/v
0:.	4?2Adam/embedding_158/embeddings/v
0:.	?2Adam/embedding_156/embeddings/v
0:.	?2Adam/embedding_157/embeddings/v
0:.	4?2Adam/embedding_161/embeddings/v
0:.	?2Adam/embedding_159/embeddings/v
0:.	?2Adam/embedding_160/embeddings/v
?2?
$__inference__wrapped_model_256359187?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

?2?
3__inference_custom_model_26_layer_call_fn_256360258
3__inference_custom_model_26_layer_call_fn_256360976
3__inference_custom_model_26_layer_call_fn_256360419
3__inference_custom_model_26_layer_call_fn_256360907?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360096
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360838
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360668
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360004?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_model_52_layer_call_fn_256361086
,__inference_model_52_layer_call_fn_256361073
,__inference_model_52_layer_call_fn_256359368
,__inference_model_52_layer_call_fn_256359413?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_model_52_layer_call_and_return_conditional_losses_256361060
G__inference_model_52_layer_call_and_return_conditional_losses_256361018
G__inference_model_52_layer_call_and_return_conditional_losses_256359290
G__inference_model_52_layer_call_and_return_conditional_losses_256359322?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_model_53_layer_call_fn_256361196
,__inference_model_53_layer_call_fn_256359639
,__inference_model_53_layer_call_fn_256361183
,__inference_model_53_layer_call_fn_256359594?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_model_53_layer_call_and_return_conditional_losses_256359516
G__inference_model_53_layer_call_and_return_conditional_losses_256359548
G__inference_model_53_layer_call_and_return_conditional_losses_256361170
G__inference_model_53_layer_call_and_return_conditional_losses_256361128?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dense_234_layer_call_fn_256361216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_234_layer_call_and_return_conditional_losses_256361207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_235_layer_call_fn_256361236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_235_layer_call_and_return_conditional_losses_256361227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_237_layer_call_fn_256361255?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_237_layer_call_and_return_conditional_losses_256361246?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_236_layer_call_fn_256361275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_236_layer_call_and_return_conditional_losses_256361266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_238_layer_call_fn_256361294?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_238_layer_call_and_return_conditional_losses_256361285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_239_layer_call_fn_256361313?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_239_layer_call_and_return_conditional_losses_256361304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_240_layer_call_fn_256361332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_240_layer_call_and_return_conditional_losses_256361323?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_241_layer_call_fn_256361351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_241_layer_call_and_return_conditional_losses_256361342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_normalize_26_layer_call_fn_256361377?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_normalize_26_layer_call_and_return_conditional_losses_256361368?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_242_layer_call_fn_256361396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_242_layer_call_and_return_conditional_losses_256361387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_256360498betscards0cards1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_52_layer_call_fn_256361407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_52_layer_call_and_return_conditional_losses_256361402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_embedding_158_layer_call_fn_256361424?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_embedding_158_layer_call_and_return_conditional_losses_256361417?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_embedding_156_layer_call_fn_256361441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_embedding_156_layer_call_and_return_conditional_losses_256361434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_embedding_157_layer_call_fn_256361458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_embedding_157_layer_call_and_return_conditional_losses_256361451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_53_layer_call_fn_256361469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_53_layer_call_and_return_conditional_losses_256361464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_embedding_161_layer_call_fn_256361486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_embedding_161_layer_call_and_return_conditional_losses_256361479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_embedding_159_layer_call_fn_256361503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_embedding_159_layer_call_and_return_conditional_losses_256361496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_embedding_160_layer_call_fn_256361520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_embedding_160_layer_call_and_return_conditional_losses_256361513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4?
$__inference__wrapped_model_256359187?.???????????LMYZST_`eflmst{|????{?x
q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

? "5?2
0
	dense_242#? 
	dense_242??????????
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360004?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "%?"
?
0?????????
? ?
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360096?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "%?"
?
0?????????
? ?
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360668?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "%?"
?
0?????????
? ?
N__inference_custom_model_26_layer_call_and_return_conditional_losses_256360838?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "%?"
?
0?????????
? ?
3__inference_custom_model_26_layer_call_fn_256360258?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "???????????
3__inference_custom_model_26_layer_call_fn_256360419?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "???????????
3__inference_custom_model_26_layer_call_fn_256360907?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "???????????
3__inference_custom_model_26_layer_call_fn_256360976?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "???????????
H__inference_dense_234_layer_call_and_return_conditional_losses_256361207^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_234_layer_call_fn_256361216QLM0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_235_layer_call_and_return_conditional_losses_256361227^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_235_layer_call_fn_256361236QST0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_236_layer_call_and_return_conditional_losses_256361266^_`0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_236_layer_call_fn_256361275Q_`0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_237_layer_call_and_return_conditional_losses_256361246]YZ/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ?
-__inference_dense_237_layer_call_fn_256361255PYZ/?,
%?"
 ?
inputs?????????
? "????????????
H__inference_dense_238_layer_call_and_return_conditional_losses_256361285^ef0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_238_layer_call_fn_256361294Qef0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_239_layer_call_and_return_conditional_losses_256361304^lm0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_239_layer_call_fn_256361313Qlm0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_240_layer_call_and_return_conditional_losses_256361323^st0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_240_layer_call_fn_256361332Qst0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_241_layer_call_and_return_conditional_losses_256361342^{|0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
-__inference_dense_241_layer_call_fn_256361351Q{|0?-
&?#
!?
inputs??????????
? "????????????
H__inference_dense_242_layer_call_and_return_conditional_losses_256361387_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
-__inference_dense_242_layer_call_fn_256361396R??0?-
&?#
!?
inputs??????????
? "???????????
L__inference_embedding_156_layer_call_and_return_conditional_losses_256361434a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_embedding_156_layer_call_fn_256361441T?/?,
%?"
 ?
inputs?????????
? "????????????
L__inference_embedding_157_layer_call_and_return_conditional_losses_256361451a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_embedding_157_layer_call_fn_256361458T?/?,
%?"
 ?
inputs?????????
? "????????????
L__inference_embedding_158_layer_call_and_return_conditional_losses_256361417a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_embedding_158_layer_call_fn_256361424T?/?,
%?"
 ?
inputs?????????
? "????????????
L__inference_embedding_159_layer_call_and_return_conditional_losses_256361496a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_embedding_159_layer_call_fn_256361503T?/?,
%?"
 ?
inputs?????????
? "????????????
L__inference_embedding_160_layer_call_and_return_conditional_losses_256361513a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_embedding_160_layer_call_fn_256361520T?/?,
%?"
 ?
inputs?????????
? "????????????
L__inference_embedding_161_layer_call_and_return_conditional_losses_256361479a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_embedding_161_layer_call_fn_256361486T?/?,
%?"
 ?
inputs?????????
? "????????????
I__inference_flatten_52_layer_call_and_return_conditional_losses_256361402X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
.__inference_flatten_52_layer_call_fn_256361407K/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_flatten_53_layer_call_and_return_conditional_losses_256361464X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
.__inference_flatten_53_layer_call_fn_256361469K/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_model_52_layer_call_and_return_conditional_losses_256359290m????9?6
/?,
"?
input_53?????????
p

 
? "&?#
?
0??????????
? ?
G__inference_model_52_layer_call_and_return_conditional_losses_256359322m????9?6
/?,
"?
input_53?????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_model_52_layer_call_and_return_conditional_losses_256361018k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
G__inference_model_52_layer_call_and_return_conditional_losses_256361060k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
,__inference_model_52_layer_call_fn_256359368`????9?6
/?,
"?
input_53?????????
p

 
? "????????????
,__inference_model_52_layer_call_fn_256359413`????9?6
/?,
"?
input_53?????????
p 

 
? "????????????
,__inference_model_52_layer_call_fn_256361073^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
,__inference_model_52_layer_call_fn_256361086^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
G__inference_model_53_layer_call_and_return_conditional_losses_256359516m????9?6
/?,
"?
input_54?????????
p

 
? "&?#
?
0??????????
? ?
G__inference_model_53_layer_call_and_return_conditional_losses_256359548m????9?6
/?,
"?
input_54?????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_model_53_layer_call_and_return_conditional_losses_256361128k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
G__inference_model_53_layer_call_and_return_conditional_losses_256361170k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
,__inference_model_53_layer_call_fn_256359594`????9?6
/?,
"?
input_54?????????
p

 
? "????????????
,__inference_model_53_layer_call_fn_256359639`????9?6
/?,
"?
input_54?????????
p 

 
? "????????????
,__inference_model_53_layer_call_fn_256361183^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
,__inference_model_53_layer_call_fn_256361196^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
K__inference_normalize_26_layer_call_and_return_conditional_losses_256361368[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? ?
0__inference_normalize_26_layer_call_fn_256361377N??+?(
!?
?
x??????????
? "????????????
'__inference_signature_wrapper_256360498?.???????????LMYZST_`eflmst{|???????
? 
???
&
bets?
bets?????????

*
cards0 ?
cards0?????????
*
cards1 ?
cards1?????????"5?2
0
	dense_242#? 
	dense_242?????????