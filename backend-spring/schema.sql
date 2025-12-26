create table if not exists country (
    id bigserial primary key,
    countryName varchar(200) not null
);

create table if not exists member (
    id bigserial primary key,
    loginId varchar(200) unique not null,
    loginPw varchar(200) not null,
    regDate timestamp not null default now(),
    updateDate timestamp not null default now(),
    name varchar(100) not null,
    email text not null,
    countryId bigint not null
);

create table if not exists board (
    id bigserial primary key,
    boardName varchar(100) not null
);

create table if not exists article (
    id bigserial primary key,
    title varchar(200) not null,
    content text not null,
    regDate timestamp not null default now(),
    updateDate timestamp not null default now(),
    boardId bigint not null,
    memberId bigint not null
);

INSERT INTO board (id, boardName) VALUES
  (1, '공지사항'),
  (2, '자유게시판'),
  (3, '질문과 답변'),
  (4, '오류사항 접수')
ON CONFLICT (id) DO NOTHING;

INSERT INTO country (id, countryName) VALUES
  (1, '한국'),
  (2, '미국'),
  (3, '일본')
ON CONFLICT (id) DO NOTHING;
